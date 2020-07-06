import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops
# import pdb
# from quantization import quantize
from topo_quant import *
from torch_geometric.nn.inits import glorot, zeros, uniform

quantize_bit = 32 # uniform
isTest = True

aggr_bit1=3
compute_bit1=3
aggr_bit2=8
compute_bit2=8

quan_layer1 = 32
quan_layer2 = 32

#cora 6487760num  weight:92302num feature:4053876num

#uniform 
#quan_bit   memery        feature         weight          acc
#32bit      207.6Mb        129.7Mb      2.95Mb          0.8250
#16bit      103.8Mb         64.8Mb      1.47Mb       0.8180
#8bit       51.9Mb          32.4Mb      0.73Mb          0.8180
#4bit       25.9Mb           16.2Mb     0.36Mb       0.8110
#2bit       12.9Mb           8.1Mb      0.18Mb      0.7930
#1bit        6.48Mb         4.1Mb       0.09Mb      0.7600

#layer-wise 6070004 417756
#quan_bit   memery        feature         weight          acc
# (4,32)    37.64Mb        21.06Mb      0.36Mb          0.8170 

#component wise
# (4,8,6,1)   (563863124, 29812104, 187008)    0.8210


class Q_GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=False, **kwargs):
        super(Q_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None):
        """
        qweight 
        alpha
        not inclued
        success!
        """
        
        if isTest:
            #print("test")
            if size is None and torch.is_tensor(x):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))         
            weight_qparams = calculate_qparams(self.weight, num_bits=quantize_bit)
            qweight = self.weight#[1433, 64]91712 [64, 7]448
            #pubmed [500, 64]32000 [64, 3]192
            #citeseer [3703, 64]238720 [64, 6]384
            qweight = quantize(self.weight, qparams=weight_qparams, flatten_dims=(1, -1), reduce_dim=None)#???
            
            pdb.set_trace()
            if torch.is_tensor(x):
                #pubmed [19717, 500]9858500 [19717, 64]1261888
                #citeseer  [3327, 3703]12319881 [3327, 64]212928
                x = quantize(x, num_bits=quantize_bit, dequantize=True)#[2708, 1433]3880564 [2708, 64]173312
                x = torch.matmul(x, qweight)
                #pubmed [19717, 64]1261888 [19717, 3]59151
                #citesser [3327, 64]212928 [3327, 6]19962
                x = quantize(x, num_bits=quantize_bit, dequantize=True)#[2708, 64]173312 [2708, 7]18956
            else:
                tmp0 = x[0]
                tmp1 = x[1]
                tmp0 = quantize(tmp0, num_bits=quantize_bit, dequantize=True)
                tmp1 = quantize(tmp1, num_bits=quantize_bit, dequantize=True)
                x = (None if x[0] is None else torch.matmul(tmp0, qweight),
                    None if x[1] is None else torch.matmul(tmp1, qweight))
                x = quantize(x, num_bits=quantize_bit, dequantize=True)
        else:
            if size is None and torch.is_tensor(x):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))

            if torch.is_tensor(x):
                x = torch.matmul(x, self.weight)
            else:
                x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                    None if x[1] is None else torch.matmul(x[1], self.weight))
        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        #pdb.set_trace()

        if isTest:
            #pubmed
            #x_i  [108365, 64]6935360 [108365, 3]325095
            #x_J  [108365, 64]6935360 [108365, 3]325095
            #citeseer
            #x_i [12431, 64]795584 [12431, 6]74586
            #x_j [12431, 64]795584 [12431, 6]74586
            x_j = quantize(x_j, num_bits=quantize_bit, dequantize=True)  # [13264, 64]848896  [13264, 1, 7]92848 
            x_i = quantize(x_i, num_bits=quantize_bit, dequantize=True)   #  [13264, 64]848896  [13264, 1, 7]92848 
            
            self.att.data = quantize(self.att.data, num_bits=quantize_bit, dequantize=True)#[1, 8, 16]  [1, 1, 14]

            x_j = x_j.view(-1, self.heads, self.out_channels)
            if x_i is None:
                alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            else:
                x_i = x_i.view(-1, self.heads, self.out_channels)
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            
            #pdb.set_trace()
            #Pubmed
            #alpha [108365, 8]866920 [108365, 1]
            #alpha [108365, 8] [108365, 1]
            #alpha [108365, 8] [108365, 1]
            #citeseer
            #[12431, 8]99448 [12431, 1]
            #[12431, 8]99448 [12431, 1]
            #[12431, 8]99448 [12431, 1]
            alpha = quantize(alpha, num_bits=quantize_bit, dequantize=True) # [13264, 8]106112  [13264, 1] 
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = quantize(alpha, num_bits=quantize_bit, dequantize=True) # [13264, 8]106112  [13264, 1]
            alpha = softmax(alpha, edge_index_i, size_i)
            alpha = quantize(alpha, num_bits=quantize_bit, dequantize=True)#[13264, 8]106112  [13264, 1]
            
            # Sample attention coefficients stochastically.
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        else:
            x_j = x_j.view(-1, self.heads, self.out_channels)
            if x_i is None:
                alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            else:
                x_i = x_i.view(-1, self.heads, self.out_channels)
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, size_i)

            # Sample attention coefficients stochastically.
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)
    
    def update(self, aggr_out):
        #pdb.set_trace()
        
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            qbias = self.bias
            pdb.set_trace()
            qbias = quantize(self.bias, num_bits=quantize_bit,flatten_dims=(-1, ))
            aggr_out = aggr_out + qbias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


"""

layer wise
"""



class Q_layerwise_GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=False, num=1,**kwargs):
        super(Q_layerwise_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num = num
        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if self.num==1:
            quan_layer = quan_layer1
        else:
            quan_layer = quan_layer2
            
        if isTest:
            #print("test")
            if size is None and torch.is_tensor(x):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))         
            weight_qparams = calculate_qparams(self.weight, num_bits=quan_layer)
            qweight = self.weight
            qweight = quantize(self.weight, qparams=weight_qparams, flatten_dims=(1, -1), reduce_dim=None)
            qweight = quantize(self.weight, num_bits=quan_layer, dequantize=True)
            
            if torch.is_tensor(x):
                
                x = quantize(x, num_bits=quan_layer, dequantize=True)
                x = torch.matmul(x, qweight)
                x = quantize(x, num_bits=quan_layer, dequantize=True)
            else:
                tmp0 = x[0]
                tmp1 = x[1]
                tmp0 = quantize(tmp0, num_bits=quan_layer, dequantize=True)
                tmp1 = quantize(tmp1, num_bits=quan_layer, dequantize=True)
                x = (None if x[0] is None else torch.matmul(tmp0, qweight),
                    None if x[1] is None else torch.matmul(tmp1, qweight))
                x = quantize(x, num_bits=quan_layer, dequantize=True)
        else:
            if size is None and torch.is_tensor(x):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))

            if torch.is_tensor(x):
                x = torch.matmul(x, self.weight)
            else:
                x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                    None if x[1] is None else torch.matmul(x[1], self.weight))
        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        #pdb.set_trace()
        if self.num==1:
            quan_layer = quan_layer1
        else:
            quan_layer = quan_layer2
        if isTest:
            x_j = quantize(x_j, num_bits=quan_layer, dequantize=True) 
            x_i = quantize(x_i, num_bits=quan_layer, dequantize=True)   
            
            self.att.data = quantize(self.att.data, num_bits=quan_layer, dequantize=True)

            x_j = x_j.view(-1, self.heads, self.out_channels)
            if x_i is None:
                alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            else:
                x_i = x_i.view(-1, self.heads, self.out_channels)
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            
            #pdb.set_trace()
            alpha = quantize(alpha, num_bits=quan_layer, dequantize=True) # 
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = quantize(alpha, num_bits=quan_layer, dequantize=True) # 
            alpha = softmax(alpha, edge_index_i, size_i)
            alpha = quantize(alpha, num_bits=quan_layer, dequantize=True)
            
            # Sample attention coefficients stochastically.
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        else:
            x_j = x_j.view(-1, self.heads, self.out_channels)
            if x_i is None:
                alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            else:
                x_i = x_i.view(-1, self.heads, self.out_channels)
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, size_i)

            # Sample attention coefficients stochastically.
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)
    
    def update(self, aggr_out):
        #pdb.set_trace()
        if self.num==1:
            quan_layer = quan_layer1
        else:
            quan_layer = quan_layer2
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            qbias = self.bias
            pdb.set_trace()
            qbias = quantize(self.bias, num_bits=quan_layer,flatten_dims=(-1, ))
            aggr_out = aggr_out + qbias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



"""
component-wise
"""
class Q_componentWise_GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=False,num=1,**kwargs):
        super(Q_componentWise_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.num = num
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if self.num == 1:
            aggr_bit = aggr_bit1
            compute_bit = compute_bit1
        else:
            aggr_bit = aggr_bit2
            compute_bit = compute_bit2
        
        if isTest:
            # pdb.set_trace()
            #print("test")
            if size is None and torch.is_tensor(x):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))         
            weight_qparams = calculate_qparams(self.weight, num_bits=compute_bit)
            qweight = self.weight
            qweight = quantize(self.weight, qparams=weight_qparams, flatten_dims=(1, -1), reduce_dim=None)
            #qweight = quantize(self.weight, num_bits=compute_bit, dequantize=True)
            
            if torch.is_tensor(x):
                
                x = quantize(x, num_bits=compute_bit, dequantize=True)
                x = torch.matmul(x, qweight)
                x = quantize(x, num_bits=compute_bit, dequantize=True)
            else:
                tmp0 = x[0]
                tmp1 = x[1]
                tmp0 = quantize(tmp0, num_bits=compute_bit, dequantize=True)
                tmp1 = quantize(tmp1, num_bits=compute_bit, dequantize=True)
                x = (None if x[0] is None else torch.matmul(tmp0, qweight),
                    None if x[1] is None else torch.matmul(tmp1, qweight))
                x = quantize(x, num_bits=compute_bit, dequantize=True)
        else:
            if size is None and torch.is_tensor(x):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))

            if torch.is_tensor(x):
                x = torch.matmul(x, self.weight)
            else:
                x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                    None if x[1] is None else torch.matmul(x[1], self.weight))
        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        #pdb.set_trace()
        if self.num == 1:
            aggr_bit = aggr_bit1
            compute_bit = compute_bit1
        else:
            aggr_bit = aggr_bit2
            compute_bit = compute_bit2

        if isTest:
            x_j = quantize(x_j, num_bits=aggr_bit, dequantize=True)  
            x_i = quantize(x_i, num_bits=aggr_bit, dequantize=True)   
            
            self.att.data = quantize(self.att.data, num_bits=compute_bit, dequantize=True)

            x_j = x_j.view(-1, self.heads, self.out_channels)
            if x_i is None:
                alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            else:
                x_i = x_i.view(-1, self.heads, self.out_channels)
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            
            #pdb.set_trace()
            alpha = quantize(alpha, num_bits=aggr_bit, dequantize=True) # 
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = quantize(alpha, num_bits=aggr_bit, dequantize=True) # !!
            alpha = softmax(alpha, edge_index_i, size_i)
            alpha = quantize(alpha, num_bits=aggr_bit, dequantize=True)
            
            # Sample attention coefficients stochastically.
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        else:
            x_j = x_j.view(-1, self.heads, self.out_channels)
            if x_i is None:
                alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            else:
                x_i = x_i.view(-1, self.heads, self.out_channels)
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, size_i)

            # Sample attention coefficients stochastically.
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)
    
    def update(self, aggr_out):
        #pdb.set_trace()
        
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            qbias = self.bias
            # pdb.set_trace()
            qbias = quantize(self.bias, num_bits=aggr_bit,flatten_dims=(-1, ))
            aggr_out = aggr_out + qbias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)









class QQ_GATConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=False,num=1, **kwargs):
        super(QQ_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num=num
        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if self.num == 1:
            aggr_bit = aggr_bit1
            compute_bit = compute_bit1
        else:
            aggr_bit = aggr_bit2
            compute_bit = compute_bit2
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))
        # pdb.set_trace()
        if torch.is_tensor(x):
            x = quantize(x, num_bits=compute_bit, dequantize=True)#[13752, 767]10547784  [13752, 64]880128
            x = torch.matmul(x, self.weight)
            x = quantize(x, num_bits=compute_bit, dequantize=True)#[13752, 64]880128 [13752, 10]137520
        else:
            tmp0 = x[0]
            tmp1 = x[1]
            tmp0 = quantize(tmp0, num_bits=compute_bit, dequantize=True)
            tmp1 = quantize(tmp1, num_bits=compute_bit, dequantize=True)
            x = (None if x[0] is None else torch.matmul(tmp0, self.weight),
                None if x[1] is None else torch.matmul(tmp1, self.weight))
            #x = quantize(x, num_bits=compute_bit, dequantize=True)
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        if self.num == 1:
            aggr_bit = aggr_bit1
            compute_bit = compute_bit1
        else:
            aggr_bit = aggr_bit2
            compute_bit = compute_bit2
        #pdb.set_trace()
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_j = quantize(x_j, num_bits=aggr_bit, dequantize=True)#[505474, 64]32350336 [505474, 10]5054740
        x_i = quantize(x_i, num_bits=aggr_bit, dequantize=True)#[505474, 64]32350336 [505474, 10]5054740

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = quantize(alpha, num_bits=aggr_bit, dequantize=True) #[505474, 8]4043792 [505474,1]
        alpha = softmax(alpha, edge_index_i, size_i)
        alpha = quantize(alpha, num_bits=aggr_bit, dequantize=True) #[505474, 8]4043792 [505474,1]

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
