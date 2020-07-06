import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from gat_conv import GATConv
from torch_geometric.datasets import Amazon

from torch_geometric.nn import AGNNConv
"""
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

"""
dataset = 'Computers'

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Amazon(path, dataset, transform=T.TargetIndegree())
data = dataset[0]

# import math
# import numpy as np

# productsCount = np.zeros(10)
# sameClassCount = np.zeros(10)
# # Iterate through the edges
# for edge_index in range(data.num_edges):
#   freqIndex = math.floor(data.edge_attr[edge_index]*10)
#   if (freqIndex == 10):
#     freqIndex = 9

#   productsCount[freqIndex] +=1
#   # If they belong in the same class 
#   if (data.y[data.edge_index[0][edge_index]] == data.y[data.edge_index[1][edge_index]]):
#     sameClassCount[freqIndex] +=1

# np.seterr(divide='ignore', invalid='ignore')
# y = np.nan_to_num(sameClassCount/productsCount)


train_split = int(data.num_nodes * 0.6)
val_split = int(data.num_nodes * 0.9)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
# data.train_mask[:data.num_nodes - 1000] = 1
data.train_mask[:train_split] = True

data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
# data.val_mask[data.num_nodes - 1000:data.num_nodes - 500] = 1
data.val_mask[train_split:val_split] = True

data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
# data.test_mask[data.num_nodes - 500:] = 1
data.test_mask[val_split:] = True

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

    def forward(self):
        
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 400):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))

    