import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import ssl
import time
import random
import os.path as osp
import argparse

from topo_quant import *

shuffle_masks = True
freq = 5
epoch_num = 200
train_prec = 0.6
val_prec = 0.9
learning_rate = 0.01

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Citeseer' # Cora, Citeseer, PubMed
# dataset = 'computers' # Amazon: computers, photo
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# dataset = Amazon(path, dataset)
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, quant=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # print(x.size())
        # print(edge_weight.size())
        # print(edge_index.size())
        # print(edge_index)
        if quant:
            x = quant_based_degree(x, edge_index)
        
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=learning_rate)

# train_mask = [False] * len(data.y)
# for i in range(int(len(data.y) * 0.6)): 
#     train_mask[i] = True
# train_mask = torch.BoolTensor(train_mask)
# train_mask = train_mask.cuda()
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

def train():
    model.train()
    optimizer.zero_grad()
    try:
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    except AttributeError:
        F.nll_loss(model()[train_mask], data.y[train_mask]).backward()
    optimizer.step()

@torch.no_grad()
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