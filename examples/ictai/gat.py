
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
import ssl
import time
import random
import os.path as osp

from topo_quant import *

shuffle_masks = True
freq = 5
epoch_num = 200
train_prec = 0.6
val_prec = 0.9
learning_rate = 0.005

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# dataset = 'Citeseer' # Cora # Citeseer # PubMed
dataset = 'photo' # amazon: computers, photo

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset) 
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures()) # Cora # Citeseer # PubMed
dataset = Amazon(path, dataset)  # amazon: computers, photo

data = dataset[0]

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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, quant=False):
        x, edge_index = data.x, data.edge_index
        if quant:
            x = quant_based_degree(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) # others
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
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #     pred = logits[mask].max(1)[1]
    #     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     accs.append(acc)

    # train_mask = [0] * len(data.y)
    # for i in range(int(len(data.y) * train_prec)): 
    #     train_mask[i] = 1

    # val_mask = [0] * len(data.y)
    # for i in range(int(len(data.y) * train_prec), int(len(data.y) * val_prec)):
    #     val_mask[i] = 1

    # test_mask = [0] * len(data.y)
    # for i in range(int(len(data.y) * val_prec), int(len(data.y) * 1.0)):
    #     test_mask[i] = 1

    # train_mask = torch.BoolTensor(train_mask).cuda()
    # val_mask = torch.BoolTensor(val_mask).cuda()
    # test_mask = torch.BoolTensor(test_mask).cuda()
    
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

