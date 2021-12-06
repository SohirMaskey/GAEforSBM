import numpy as np

import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.nn import GAE

import torch.nn.functional as F

import os.path as osp
import os

class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        #super().__init__()
        super(GCN, self).__init__()
        self.lin1 = torch.nn.Linear(1, 16)
        self.lin2 = torch.nn.Linear(16, 1)
        self.graphSage = SAGEConv(num_node_features, 2, root_weight=False, bias=False)

        #self.graphSage = SAGEConv(DL.get(101).num_node_features,1 ,root_weight = False, bias = False)
        
    def forward(self, x, edge_index):
        x, edge_index = x, edge_index

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        #x = F.dropout(x, training=self.training)
        x = self.graphSage(x, edge_index)

        return x
        #F.log_softmax(x, dim=1)

        
if __name__ == "__main__":
    graph = torch.load(osp.join('../input/', 'RGG_1000nodes.pt'))
    model = GAE(GCN(graph.num_node_features))
    torch.save(model.state_dict(), '../models/TwoLayerGraphSage.pt')