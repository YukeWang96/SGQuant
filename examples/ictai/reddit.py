import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv, AGNNConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BatchNorm

from topo_quant import *

freq = 5
epo_num = 6
GCN = True
GIN = False
GAT = False
Cond = GCN + GIN + GAT
assert Cond == 1
hidden = 16
head = 8
bit_list = [32,32,32,32]

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

if GIN:
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[10, 10, 10, 10, 10], batch_size=64, shuffle=True,
                                num_workers=16)
if GCN or GIN:
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                            sizes=[25, 10], batch_size=1024, shuffle=True,
                            num_workers=16)

subgraph_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=16)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        # self.convs.append(SAGEConv(in_channels, hidden_channels))
        # self.convs.append(SAGEConv(hidden_channels, out_channels))
        if GCN:
            self.num_layers = 2
            # self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
            # self.convs.append(GCNConv(hidden_channels, out_channels, normalize=True))
            
            self.lin1 = torch.nn.Linear(dataset.num_features, 16)
            # self.prop1 = AGNNConv(requires_grad=False)
            self.convs.append(AGNNConv(requires_grad=False))
            # self.prop2 = AGNNConv(requires_grad=True)
            self.convs.append(AGNNConv(requires_grad=True))
            self.lin2 = torch.nn.Linear(16, dataset.num_classes)

        if GIN:
            self.num_layers = 5
            self.batch_norms = torch.nn.ModuleList()

            for i in range(self.num_layers):
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

        if GAT:
            self.num_layers = 2
            self.convs.append(GATConv(in_channels, hidden_channels, heads=head, dropout=0.6))
            self.convs.append(GATConv(hidden_channels * head, out_channels, heads=1, concat=True, dropout=0.6))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        x = quantize(x, num_bits=bit_list[0], dequantize=True)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))

        for i, (edge_index, _, size) in enumerate(adjs):

            # print(size)
            # x_target = x[:size[1]]  # Target nodes are always placed first.
            # x = self.convs[i]((x, x_target), edge_index)
            if i == 0 and GAT:
                x = F.dropout(x, p=0.6, training=self.training)
                
            if GCN or GAT:

                # x = quantize(x, num_bits=bit_list[1], dequantize=True)
                # x = self.prop1(x, data.edge_index)
                # x = quantize(x, num_bits=bit_list[2], dequantize=True)
                # x = self.prop2(x, data.edge_index)
                x = quantize(x, num_bits=bit_list[i+1], dequantize=True)
                x = self.convs[i](x, edge_index)
                # print("x_target.size(): ", x_target.size())
                # print("x.size(): ", x.size())
                # print(edge_index)
                # print(edge_index.size())

                # if i != self.num_layers - 1:
                #     x = F.relu(x)
                #     if GCN:
                #         x = F.dropout(x, training=self.training)
                #     else:
                #         x = F.dropout(x, p=0.6, training=self.training)
            if GIN:
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(self.batch_norms[i](x))
        
        if GCN:
            x = F.dropout(x, training=self.training)
            x = quantize(x, num_bits=bit_list[3], dequantize=True)
            x = self.lin2(x)

        if GAT:
            x = F.dropout(x, training=self.training)
            x = quantize(x, num_bits=bit_list[3], dequantize=True)
            x = self.lin2(x)

        if GIN:
            # x = global_add_pool(x, batch)
            x = F.relu(self.batch_norm1(self.lin1(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)

        return x.log_softmax(dim=-1)

    def inference(self, x_all, quant=False):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar = tqdm(total= len(subgraph_loader) * self.num_layers)
        pbar.set_description('Evaluating')
        xs = []
        
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        if GCN or GAT:
            # for i in range(self.num_layers):
            #     for batch_size, n_id, adj in subgraph_loader:
            #         edge_index, _, size = adj.to(device)
            #         x = x_all[n_id].to(device)
            #         # x_target = x[:size[1]]
            #         # x = self.convs[i]((x, x_target), edge_index)
            #         if quant:
            #             x = quant_based_degree(x, edge_index)

            #         x = self.convs[i](x, edge_index)
            #         if i != self.num_layers - 1:
            #             x = F.relu(x)
            #         # xs.append(x[:size[1]].cpu())
            #         xs.append(x.cpu())
            #         pbar.update(batch_size)

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                # x_target = x[:size[1]]
                # x = self.convs[i]((x, x_target), edge_index)
                

                for i in range(self.num_layers):
                    if quant:
                        x = quant_based_degree(x, edge_index, layer_num=i+1)

                    x = self.convs[i](x, edge_index)
                    if i != self.num_layers - 1:
                        x = F.relu(x)

                xs.append(x.cpu())
                pbar.update(self.num_layers)    
            
            # x_all = torch.cat(xs, dim=0)
            pbar.close()
        
        if GIN:
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                for conv, batch_norm in zip(self.convs, self.batch_norms):
                    x = F.relu(batch_norm(conv(x, edge_index)))
                # x = global_add_pool(x, batch)
                x = F.relu(self.batch_norm1(self.lin1(x)))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.lin2(x)
                xs.append(x)
                pbar.update(batch_size)

            # x_all = torch.cat(xs, dim=0)
            pbar.close()

        # return x_all
        return xs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, hidden, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    total_nds = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        # print("x[n_id].size() ", x[n_id].size())
        out = model(x[n_id], adjs)
        
        # print("x[n_id].size() ", x[n_id].size())
        # loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss = F.nll_loss(out, y[n_id])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        # total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        total_correct += int(out.argmax(dim=-1).eq(y[n_id]).sum())
        total_nds += len(n_id)
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / total_nds # int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(quant=False):
    model.eval()

    out_list = model.inference(x, quant=quant)
    y_true = y.cpu().unsqueeze(-1)
    # y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    # for mask in [data.train_mask, data.val_mask, data.test_mask]:
    #     results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    if GCN or GAT:
        begin = 0
        for batch_size, n_id, adj in subgraph_loader:
            out = out_list.pop(0)
            y_pred = out.argmax(dim=-1, keepdim=True)
            _, _, size = adj.to(device)
            results.append(float(y_pred.eq(y_true[n_id]).sum()) / len(n_id))
    if GIN:
        for batch_size, n_id, adj in subgraph_loader:
            out = out_list.pop(0)
            y_pred = out.argmax(dim=-1, keepdim=True)
            _, _, size = adj.to(device)
            results.append(float(y_pred.eq(y_true[n_id]).sum()) / len(n_id))

    return sum(results)/len(results)


for epoch in range(1, epo_num + 1):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    # train_acc, val_acc, test_acc = test()
    # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #       f'Test: {test_acc:.4f}')
    test_acc = test()
    print(f'Test: {test_acc:.4f}')

    if epoch != 1 and (epoch - 1) % freq == 0:
        test_acc_quant = test(quant=True)
        print(f'Quant_Test: {test_acc_quant:.4f}')
    

# import os.path as osp

# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from torch_geometric.datasets import Reddit
# from torch_geometric.data import NeighborSampler
# from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv, AGNNConv
# from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BatchNorm

# from topo_quant import *

# freq = 5
# epo_num = 6
# GCN = True
# GIN = False
# GAT = False
# Cond = GCN + GIN + GAT
# assert Cond == 1
# hidden = 32
# head = 8
# bit_list = [32,32,32,32]

# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]

# if GIN:
#     train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
#                                 sizes=[10, 10, 10, 10, 10], batch_size=64, shuffle=True,
#                                 num_workers=16)
# if GCN or GIN:
#     train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
#                             sizes=[25, 10], batch_size=1024, shuffle=True,
#                             num_workers=16)

# subgraph_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[-1],
#                                   batch_size=1024, shuffle=False,
#                                   num_workers=16)


# class SAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(SAGE, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         # self.convs.append(SAGEConv(in_channels, hidden_channels))
#         # self.convs.append(SAGEConv(hidden_channels, out_channels))
#         if GCN:
#             self.num_layers = 2
#             # self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
#             # self.convs.append(GCNConv(hidden_channels, out_channels, normalize=True))
            
#             self.lin1 = torch.nn.Linear(dataset.num_features, 16)
#             # self.prop1 = AGNNConv(requires_grad=False)
#             self.convs.append(AGNNConv(requires_grad=False))
#             # self.prop2 = AGNNConv(requires_grad=True)
#             self.convs.append(AGNNConv(requires_grad=True))
#             self.lin2 = torch.nn.Linear(16, dataset.num_classes)

#         if GIN:
#             self.num_layers = 5
#             self.batch_norms = torch.nn.ModuleList()

#             for i in range(self.num_layers):
#                 mlp = Sequential(
#                     Linear(in_channels, 2 * hidden_channels),
#                     BatchNorm(2 * hidden_channels),
#                     ReLU(),
#                     Linear(2 * hidden_channels, hidden_channels),
#                 )
#                 conv = GINConv(mlp, train_eps=True)
#                 self.convs.append(conv)
#                 self.batch_norms.append(BatchNorm(hidden_channels))

#                 in_channels = hidden_channels
                
#             self.lin1 = Linear(hidden_channels, hidden_channels)
#             self.batch_norm1 = BatchNorm(hidden_channels)
#             self.lin2 = Linear(hidden_channels, out_channels)

#         if GAT:
#             self.num_layers = 2
#             self.convs.append(GATConv(in_channels, hidden_channels, heads=head, dropout=0.6))
#             self.convs.append(GATConv(hidden_channels * head, out_channels, heads=1, concat=True, dropout=0.6))

#     def forward(self, x, adjs):
#         # `train_loader` computes the k-hop neighborhood of a batch of nodes,
#         # and returns, for each layer, a bipartite graph object, holding the
#         # bipartite edges `edge_index`, the index `e_id` of the original edges,
#         # and the size/shape `size` of the bipartite graph.
#         # Target nodes are also included in the source nodes so that one can
#         # easily apply skip-connections or add self-loops.
#         for i, (edge_index, _, size) in enumerate(adjs):

#             # print(size)
#             # x_target = x[:size[1]]  # Target nodes are always placed first.
#             # x = self.convs[i]((x, x_target), edge_index)
#             if i == 0 and GAT:
#                 x = F.dropout(x, p=0.6, training=self.training)
            
#             if GCN:
#                 x = quantize(x, num_bits=bit_list[0], dequantize=True)

#                 x = F.dropout(x, training=self.training)
#                 x = F.relu(self.lin1(x))
#                 x = quantize(x, num_bits=bit_list[1], dequantize=True)
#                 x = self.prop1(x, data.edge_index)
#                 x = quantize(x, num_bits=bit_list[2], dequantize=True)
#                 x = self.prop2(x, data.edge_index)
#                 x = F.dropout(x, training=self.training)
#                 x = quantize(x, num_bits=bit_list[3], dequantize=True)
#                 x = self.lin2(x)

#             if GCN or GAT:

#                 # x = quantize(x, num_bits=bit_list[1], dequantize=True)
#                 # x = self.prop1(x, data.edge_index)
#                 # x = quantize(x, num_bits=bit_list[2], dequantize=True)
#                 # x = self.prop2(x, data.edge_index)
#                 x = quantize(x, num_bits=bit_list[i+1], dequantize=True)
#                 x = self.convs[i](x, edge_index)
#                 # print("x_target.size(): ", x_target.size())
#                 # print("x.size(): ", x.size())
#                 # print(edge_index)
#                 # print(edge_index.size())

#                 # if i != self.num_layers - 1:
#                 #     x = F.relu(x)
#                 #     if GCN:
#                 #         x = F.dropout(x, training=self.training)
#                 #     else:
#                 #         x = F.dropout(x, p=0.6, training=self.training)
#             if GIN:
#                 x = self.convs[i](x, edge_index)
#                 if i != self.num_layers - 1:
#                     x = F.relu(self.batch_norms[i](x))
        
#         if GAT:
#             x = F.dropout(x, training=self.training)
#             x = quantize(x, num_bits=bit_list[3], dequantize=True)
#             x = self.lin2(x)

#         if GIN:
#             # x = global_add_pool(x, batch)
#             x = F.relu(self.batch_norm1(self.lin1(x)))
#             x = F.dropout(x, p=0.5, training=self.training)
#             x = self.lin2(x)

#         return x.log_softmax(dim=-1)

#     def inference(self, x_all, quant=False):
#         # pbar = tqdm(total=x_all.size(0) * self.num_layers)
#         pbar = tqdm(total= len(subgraph_loader) * self.num_layers)
#         pbar.set_description('Evaluating')
#         xs = []


#         # Compute representations of nodes layer by layer, using *all*
#         # available edges. This leads to faster computation in contrast to
#         # immediately computing the final representations of each batch.
#         if GCN or GAT:
#             # for i in range(self.num_layers):
#             #     for batch_size, n_id, adj in subgraph_loader:
#             #         edge_index, _, size = adj.to(device)
#             #         x = x_all[n_id].to(device)
#             #         # x_target = x[:size[1]]
#             #         # x = self.convs[i]((x, x_target), edge_index)
#             #         if quant:
#             #             x = quant_based_degree(x, edge_index)

#             #         x = self.convs[i](x, edge_index)
#             #         if i != self.num_layers - 1:
#             #             x = F.relu(x)
#             #         # xs.append(x[:size[1]].cpu())
#             #         xs.append(x.cpu())
#             #         pbar.update(batch_size)

#             for batch_size, n_id, adj in subgraph_loader:
#                 edge_index, _, size = adj.to(device)
#                 x = x_all[n_id].to(device)
#                 # x_target = x[:size[1]]
#                 # x = self.convs[i]((x, x_target), edge_index)
                

#                 for i in range(self.num_layers):
#                     if quant:
#                         x = quant_based_degree(x, edge_index, layer_num=i+1)

#                     x = self.convs[i](x, edge_index)
#                     if i != self.num_layers - 1:
#                         x = F.relu(x)

#                 xs.append(x.cpu())
#                 pbar.update(self.num_layers)    
            
#             # x_all = torch.cat(xs, dim=0)
#             pbar.close()
        
#         if GIN:
#             for batch_size, n_id, adj in subgraph_loader:
#                 edge_index, _, size = adj.to(device)
#                 x = x_all[n_id].to(device)
#                 for conv, batch_norm in zip(self.convs, self.batch_norms):
#                     x = F.relu(batch_norm(conv(x, edge_index)))
#                 # x = global_add_pool(x, batch)
#                 x = F.relu(self.batch_norm1(self.lin1(x)))
#                 x = F.dropout(x, p=0.5, training=self.training)
#                 x = self.lin2(x)
#                 xs.append(x)
#                 pbar.update(batch_size)

#             # x_all = torch.cat(xs, dim=0)
#             pbar.close()

#         # return x_all
#         return xs


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SAGE(dataset.num_features, hidden, dataset.num_classes)
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# x = data.x.to(device)
# y = data.y.squeeze().to(device)


# def train(epoch):
#     model.train()

#     pbar = tqdm(total=int(data.train_mask.sum()))
#     pbar.set_description(f'Epoch {epoch:02d}')

#     total_loss = total_correct = 0
#     total_nds = 0
#     for batch_size, n_id, adjs in train_loader:
#         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
#         adjs = [adj.to(device) for adj in adjs]

#         optimizer.zero_grad()
#         # print("x[n_id].size() ", x[n_id].size())
#         out = model(x[n_id], adjs)
        
#         # print("x[n_id].size() ", x[n_id].size())
#         # loss = F.nll_loss(out, y[n_id[:batch_size]])
#         loss = F.nll_loss(out, y[n_id])
#         loss.backward()
#         optimizer.step()

#         total_loss += float(loss)
#         # total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
#         total_correct += int(out.argmax(dim=-1).eq(y[n_id]).sum())
#         total_nds += len(n_id)
#         pbar.update(batch_size)

#     pbar.close()

#     loss = total_loss / len(train_loader)
#     approx_acc = total_correct / total_nds # int(data.train_mask.sum())

#     return loss, approx_acc


# @torch.no_grad()
# def test(quant=False):
#     model.eval()

#     out_list = model.inference(x, quant=quant)
#     y_true = y.cpu().unsqueeze(-1)
#     # y_pred = out.argmax(dim=-1, keepdim=True)

#     results = []
#     # for mask in [data.train_mask, data.val_mask, data.test_mask]:
#     #     results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
#     if GCN or GAT:
#         begin = 0
#         for batch_size, n_id, adj in subgraph_loader:
#             out = out_list.pop(0)
#             y_pred = out.argmax(dim=-1, keepdim=True)
#             _, _, size = adj.to(device)
#             results.append(float(y_pred.eq(y_true[n_id]).sum()) / len(n_id))
#     if GIN:
#         for batch_size, n_id, adj in subgraph_loader:
#             out = out_list.pop(0)
#             y_pred = out.argmax(dim=-1, keepdim=True)
#             _, _, size = adj.to(device)
#             results.append(float(y_pred.eq(y_true[n_id]).sum()) / len(n_id))

#     return sum(results)/len(results)


# for epoch in range(1, epo_num + 1):
#     loss, acc = train(epoch)
#     print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
#     # train_acc, val_acc, test_acc = test()
#     # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
#     #       f'Test: {test_acc:.4f}')
#     test_acc = test()
#     print(f'Test: {test_acc:.4f}')

#     if epoch != 1 and (epoch - 1) % freq == 0:
#         test_acc_quant = test(quant=True)
#         print(f'Quant_Test: {test_acc_quant:.4f}')
    