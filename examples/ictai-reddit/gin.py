
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BatchNorm
import ssl
import time
import os.path as osp
import random

from topo_quant import *

shuffle_masks = True
epoch_num = 200
train_prec = 0.6
val_prec = 0.9
freq = 5
learning_rate = 0.01

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

##############################################################################
dataset = 'Cora' # Cora # Citeseer # PubMed
# dataset = 'photo' # amazon: computers, photo

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# dataset = Amazon(path, dataset)

data = dataset[0]
###############################################################################

train_mask = [0] * len(data.y)
for i in range(int(len(data.y) * train_prec)): 
    train_mask[i] = 1

val_mask = [0] * len(data.y)
for i in range(int(len(data.y) * train_prec), int(len(data.y) * val_prec)):
    val_mask[i] = 1

test_mask = [0] * len(data.y)
for i in range(int(len(data.y) * val_prec), int(len(data.y) * 1.0)):
    test_mask[i] = 1

if shuffle_masks:
    random.shuffle(train_mask)
    random.shuffle(val_mask)
    random.shuffle(test_mask)

train_mask = torch.BoolTensor(train_mask).cuda()
val_mask = torch.BoolTensor(val_mask).cuda()
test_mask = torch.BoolTensor(test_mask).cuda()
###############################################################################

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_channels = dataset.num_features
        hidden_channels = 64
        out_channels = dataset.num_classes
        num_layers = 5

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, quant=False):
        x, edge_index = data.x, data.edge_index
        if quant:
            x = quant_based_degree(x, edge_index)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        # x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # others
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4) # amazon

def train():
    model.train()
    optimizer.zero_grad()
    try:
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    except AttributeError:
        F.nll_loss(model()[train_mask], data.y[train_mask]).backward()
    optimizer.step()


def test(quant=False):
    global train_mask
    global val_mask
    global test_mask

    model.eval()
    logits, accs = model(quant), []

    # print(train_mask)
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        tmp = torch.eq(pred, data.y[mask])
        correct = sum(tmp).cpu().item()
        total = sum(mask).cpu().item()
        acc = correct/total
        accs.append(acc)
    return accs

# for epoch in range(1, epoch_num + 1):
#     train()
#     log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#     print(log.format(epoch, *test()))
#     if epoch % freq == 0:
#         print("* quant ", log.format(epoch, *test(True)))

best_test_acc = 0
best_qnt_test_acc = 0
best_val_acc = test_acc = 0
for epoch in range(1, epoch_num + 1):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    best_test_acc = max(tmp_test_acc, best_test_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    
    if epoch % freq == 0:
        quant_train, quant_val, quant_test = test(True)
        print("==> quant ", log.format(epoch, quant_train, quant_val, quant_test))
        best_qnt_test_acc = max(best_qnt_test_acc, quant_test)
    

print("\n\n")
print("**best_test_acc:\t{:.4f}\n**best_qnt_test_acc:\t{:.4f}".format(best_test_acc, best_qnt_test_acc))