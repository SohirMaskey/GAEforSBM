import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F

import os.path as osp
import os

from TwoLayerGraphSage import GCN 

import matplotlib.pyplot as plt

graph = torch.load(osp.join('../input/', 'RGG_1000nodes.pt'))
data = train_test_split_edges(graph)

train_pos_edge_index = data.train_pos_edge_index

"""
gS = SAGEConv(graph.num_node_features, 2, root_weight=False, bias=False)
model = GAE(gS)

model.load_state_dict(torch.load( '../models/GAESage'))
"""

model = GAE(GCN(graph.num_node_features ))
model.load_state_dict(torch.load( '../models/TwoLayerGraphSage.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_pos_edge_index = train_pos_edge_index.to(device)
graph = graph.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(graph.x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    print(loss)
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(graph.x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

epochs = 100

for epoch in range(1, epochs+1):
    loss = train()

    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
torch.save(model.state_dict(), '../models/' + str(epochs) + 'EpochsTrainedTwoLayerGraphSage.pt')




