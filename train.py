import os, sys, datetime, random, traceback, warnings, argparse, logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# trochs
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel

from dataSets.datasetGFFNet import build_loader
from models.GFFNet import Network
from metrics import Metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def log_acc(logger, metrics):
    logger.info(metrics.confusion_matrix())
    logger.info(f"TP@0: {metrics.TP(0)}, FP@0: {metrics.FP(0)}, TN@0: {metrics.TN(0)}, FN@0: {metrics.FN(0)}")
    logger.info(f"TP@1: {metrics.TP(1)}, FP@1: {metrics.FP(1)}, TN@1: {metrics.TN(1)}, FN@1: {metrics.FN(1)}")
    logger.info(f"TP@2: {metrics.TP(2)}, FP@2: {metrics.FP(2)}, TN@2: {metrics.TN(2)}, FN@2: {metrics.FN(2)}")
    
    logger.info(f"acc: {metrics.acc()}")
    logger.info(f"acc@0: {metrics.acc(0)}, acc@1: {metrics.acc(1)}, acc@2: {metrics.acc(2)}")
    logger.info(f"sensitivity@0: {metrics.sensitivity(0)}, sensitivity@1: {metrics.sensitivity(1)}, sensitivity@2: {metrics.sensitivity(2)}")
    logger.info(f"specificity@0: {metrics.specificity(0)}, specificity@1: {metrics.specificity(1)}, specificity@2: {metrics.specificity(2)}")

    logger.info(f"recall: {metrics.recall()}")
    logger.info(f"recall@0: {metrics.recall(0)}, recall@1: {metrics.recall(1)}, recall@2: {metrics.recall(2)}")
    logger.info(f"precision: {metrics.precision()}")
    logger.info(f"precision@0: {metrics.precision(0)}, precision@1: {metrics.precision(1)}, precision@2: {metrics.precision(2)}")
    
    logger.info(f"f1: {metrics.f1()}")
    logger.info(f"f1@0: {metrics.f1(0)}, f1@1: {metrics.f1(1)}, f1@2: {metrics.f1(2)}")
    logger.info(f"kappa: {metrics.kappa()}")

    tpr, fpr, auc_normal = metrics.AUC_ROC(0)
    tpr, fpr, auc_plus = metrics.AUC_ROC(2)
    logger.info(f"AUC@0: {auc_normal}, AUC@2: {auc_plus}")

    return


def train(model, data_loader, criterion, optimizer, alpha):
    model.train()
    correct, total_loss, total = 0, 0, 0
    time_start = datetime.datetime.now()
    
    for data in tqdm(data_loader):
        labels, imgs, edge_index, exam_name = data
        labels = labels.cuda().long()
        
        optimizer.zero_grad()
        outputs, aux_outputs = model(imgs, edge_index) # outputs size: [batchSzie, classNum]
        main_loss = criterion(outputs, labels)
        aux_loss = criterion(aux_outputs, labels)
        loss = alpha * main_loss + (1 - alpha) * aux_loss
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        total_loss += loss

        _, predicted = torch.max(outputs.detach(), 1)
        correct += torch.sum(predicted.detach() == labels.detach())

    acc = float(correct) / total
    loss = total_loss / len(data_loader)
    time_spend = (datetime.datetime.now() - time_start).seconds

    return acc, loss, time_spend


@torch.no_grad()
def test(model, data_loader, criterion, alpha):
    model.eval()
    metrics = Metrics(3)
    correct, total_loss, total = 0, 0, 0
    time_start = datetime.datetime.now() 
    
    for data in tqdm(data_loader):
        labels, imgs, edge_index, exam_name = data
        labels = labels.cuda().long()
        
        outputs, aux_outputs = model(imgs, edge_index) # outputs size: [batchSzie, classNum]
        main_loss = criterion(outputs, labels)
        aux_loss = criterion(aux_outputs, labels)
        loss = alpha * main_loss + (1 - alpha) * aux_loss
        
        total_loss += loss
        total += labels.size(0)

        _, predicted = torch.max(outputs.detach(), 1)
        correct += torch.sum(predicted.detach() == labels.detach())

        softmax = F.softmax(outputs, dim=1).detach().cpu().numpy()
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()
        for i, predict in enumerate(predicted):
            label = labels[i]
            scores = softmax[i]
            metrics.add_sample(predict, label, scores)

    acc = float(correct) / total
    loss = total_loss / len(data_loader)
    
    time_spend = (datetime.datetime.now() - time_start).seconds
    return acc, loss, time_spend, metrics


def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True # this item has impact on the deterministic results when the optimizers are adaptive ones
    torch.backends.cudnn.benchmark = False  

    output_path = os.path.join("exps", args.log_name)
    os.makedirs(output_path, exist_ok=True)

    log_path = os.path.join(output_path, 'log.log')
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(levelname)s:%(name)s:%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    
    checkpoint_path = os.path.join(output_path, "ckpts")
    os.makedirs(checkpoint_path, exist_ok=True)

    logger.info(args)

    # datasets
    train_loader, test_loader, data_weight = build_loader(args)
    logger.info("Iters train: {}, test: {}".format(len(train_loader), len(test_loader)))

    # model
    model = Network(args.graph_conv)
    model = model.cuda()
    model = DataParallel(model)

    # optimizer
    param_group = [
        {'params': model.module.channel_conv.parameters()},
        {'params': model.module.backbone.parameters(), 'lr': 1e-3 / 10},
        {'params': model.module.graph_feature_fusion.parameters()},
        {'params': model.module.head.parameters()},
        {'params': model.module.aux_head.parameters()},
    ]
    optimizer = optim.AdamW(param_group, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    logger.info(f"CE Loss data weight: {data_weight}")
    criterion = nn.CrossEntropyLoss(weight=data_weight).cuda() # cls loss

    best_test_acc, bset_epoch = 0, 0

    ## train ##
    for epoch in range(args.epoch):

        # log learning rate
        lr = optimizer.state_dict().get("param_groups")[0].get("lr")
        logger.info("Epoch {}, LR: {}".format(epoch, lr))

        ## train
        acc, loss, time_spend = train(model, train_loader, criterion, optimizer, args.alpha)
        logger.info("Epoch {}, Train time: {}, Train loss: {:.6f}, Train acc: {:.6f}.".format(epoch, time_spend, loss, acc))
        scheduler.step(loss) # train_loss val_loss

        # test per TEST_PER_EPOCH epoch
        test_acc, loss, time_spend, metrics = test(model, test_loader, criterion, args.alpha)
        logger.info("Epoch {}, Test time: {}, Test loss: {:.6f}, Test acc: {:.6f}.".format(epoch, time_spend, loss, test_acc))
        roc_dict = log_acc(logger, metrics)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            bset_epoch = epoch
            ckpt_path = os.path.join(checkpoint_path, f"epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Test ACC increased to {test_acc}") 
        else:
            logger.info(f"Test ACC not increase, best ACC is {best_test_acc} in epoch {bset_epoch}") 

    ## save best model
    logger.info("Model Train Success !")
    logger.info(f"Best test acc is {best_test_acc} in epoch {bset_epoch}")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser('training script', add_help=False)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--log_name', type=str, default="gff_net")
    parser.add_argument('--graph_conv', type=str, default="SAGE")

    parser.add_argument('--posterior_path', type=str, default=None)
    parser.add_argument('--peripherl_path', type=str, default=None)
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)

    args = parser.parse_args()

    main(args)






