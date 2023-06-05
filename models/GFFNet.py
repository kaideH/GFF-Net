# trochs
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# pyg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TopKPooling, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data, DataLoader as PygDataLoader

from torchcam.methods import SmoothGradCAMpp

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class GraphFeatureFusion(nn.Module):
    def __init__(self, graph, feature_size):
        super().__init__()

        if graph == "SAGE":
            graph_func = SAGEConv
        elif graph == "GCN":
            graph_func = GCNConv
        elif graph == "GAT":
            graph_func = GATConv

        self.conv1 = graph_func(feature_size, 1024)
        self.pool1 = TopKPooling(1024, ratio=0.8)
        self.conv2 = graph_func(1024, 1024)
        self.pool2 = TopKPooling(1024, ratio=0.8)
        self.conv3 = graph_func(1024, 1024)
        self.pool3 = TopKPooling(1024, ratio=0.8)
    
    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MaxFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = []
        for img in x:
            out.append(torch.max(img, 0)[0])
        out = torch.stack(out, 0) # [B, C]
        return out


class Network(nn.Module):
    def __init__(self, graph_conv):
        super().__init__()
        
        # feature extraction backbone
        self.channel_conv = BasicConv2d(1, 3, kernel_size=1)
        self.backbone = nn.Sequential(*list(torchvision.models.densenet169(pretrained=True).children())[:-1])
        
        # graph feature fusion
        self.graph_feature_fusion = GraphFeatureFusion(graph_conv, 1664)

        # head
        self.head = nn.Linear(2048, 3)

        # aux head
        self.aux_fusion = MaxFusion()
        self.aux_head = nn.Linear(1664, 3)
    
    def forward(self, x, batch_edge_index):
        batch_size = len(x)
        img_len = []
        for img in x:
            img_len.append(img.shape[0])

        # feature extraction
        x = torch.cat(x, 0)
        x = self.channel_conv(x)
        x = self.backbone(x).squeeze() # [B*N, 1, H, W] -> [B*N, 2048]
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = x.split(img_len, 0) # [B*N, 2048] -> [B, N, 2048]
        
        # aux fc
        aux_out = self.aux_fusion(x)
        aux_out = self.aux_head(aux_out)

        # graph feature fusion
        dataset = []
        for idx, graph in enumerate(x):
            e_index = torch.tensor(batch_edge_index[idx]).cuda()
            dataset.append(Data(x=graph, edge_index=e_index))
        data_loder = PygDataLoader(dataset, batch_size=batch_size)
        batch = next(iter(data_loder))
        batch.cuda()
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        x = self.graph_feature_fusion(x, edge_index, batch)

        # fc
        out = self.head(x)

        return out, aux_out


