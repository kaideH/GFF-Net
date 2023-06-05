import os, time, random, numbers
import numpy as np
import torch
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader


class DataSet(torch.utils.data.Dataset):
    def __init__(self, args, transforms, data_txt):
        self.transforms = transforms
        self.posterior_path = args.posterior_path
        self.peripheral_path = args.peripherl_path

        self.exams = []
        self.label_count = {}
        
        # 加载数据
        self._loadData(data_txt)
        
    def _loadData(self, data_txt):
        f = open(data_txt)
        lines = f.readlines()
        f.close()
    
        for line in lines:
            label, exam_name = line.strip().split(",")
            posterior_path = os.path.join(self.posterior_path, exam_name + ".npy")
            peripheral_path = os.path.join(self.peripheral_path, exam_name + ".npy")

            posterior = list(np.load(posterior_path)) if os.path.exists(posterior_path) else []
            peripheral = list(np.load(peripheral_path)) if os.path.exists(peripheral_path) else []
            imgs = posterior + peripheral
            if len(imgs) <= 1:
                continue
                
            posterior_idx = list(range(len(posterior)))
            peripheral_idx = list(range(len(posterior), len(posterior) + len(peripheral)))

            # step 2: save img and edge index
            self.exams.append([imgs, posterior_idx, peripheral_idx, exam_name, int(label)])
            self.label_count.setdefault(int(label), 0)
            self.label_count[int(label)] += 1
    
    def __len__(self):
        return len(self.exams)

    def data_weight(self):
        total = sum(self.label_count.values())
        weight = []
        for label in self.label_count.keys():
            weight.append((total - self.label_count[label]) / total)
        return torch.tensor(weight)

    def _get_edge_index(self, center_idx, other_idx):
        edge_index = [ [], [] ]
        
        for i in center_idx:
            for j in center_idx:
                if i == j:
                    continue
                edge_index[0].append(i)
                edge_index[1].append(j)
        
        for i in center_idx:
            for j in other_idx:
                if i == j:
                    print("exception: center and other have same idx", center_idx, other_idx)
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)
        
        if len(center_idx) == 0:
            for i in other_idx:
                for j in other_idx:
                    if i == j:
                        continue
                    edge_index[0].append(i)
                    edge_index[1].append(j)

        return edge_index

    def __getitem__(self, idx):
        # step 1: 读取图像数据
        imgs, posterior_idx, peripheral_idx, exam_name, label = self.exams[idx]

        # step: 根据图像关系确定图 edge index
        edge_index = self._get_edge_index(posterior_idx, peripheral_idx)
        
        imgs = self.transforms(imgs)

        return label, imgs, edge_index, exam_name


def collate_batch(batch_list):
    label, imgs, edge_index, exam_name = list(zip(*batch_list))
    return torch.tensor(label), imgs, edge_index, exam_name


def build_loader(args):
    t = transforms.Compose([
        Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])),
    ])
    dataset = DataSet(args, t, "/root/workspace/ROP/data_list/split_0.txt")
    data_weight = dataset.data_weight()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    t = transforms.Compose([
        Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])),
    ])
    dataset = DataSet(args, t, "/root/workspace/ROP/data_list/split_1.txt")
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader, data_weight

