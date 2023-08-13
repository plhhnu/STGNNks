##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from sklearn import metrics
from scipy import sparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader



def get_graph(adj, X):
    # create sparse matrix
    row_col = []
    edge_weight = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz() 
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)  
    graph_bags.append(graph)
    return graph_bags


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)
        
        self.prelu = nn.PReLU(hidden_channels)
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)



def train_DGI(args, data_loader, in_channels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-6)
    if args.load:
        DGI_filename = args.model_path+'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(args.num_epoch) + '.pth.tar'
        DGI_model.load_state_dict(torch.load(DGI_filename))
    else:
        import datetime
        start_time = datetime.datetime.now()
        for epoch in range(args.num_epoch):
            DGI_model.train()
            DGI_optimizer.zero_grad()
        
            DGI_all_loss = []
            
            for data in data_loader:
                data = data.to(device)
                pos_z, neg_z, summary = DGI_model(data=data)

                DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
                DGI_loss.backward()
                DGI_all_loss.append(DGI_loss.item())
                DGI_optimizer.step()

            if ((epoch+1)%100) == 0:
                print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))

        end_time = datetime.datetime.now()
        DGI_filename =  args.model_path+'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(args.num_epoch) + '.pth.tar'
        torch.save(DGI_model.state_dict(), DGI_filename)
        print('Training time in seconds: ', (end_time-start_time).seconds)
    return DGI_model


