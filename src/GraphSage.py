import numpy as np

import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.nn import GAE

import os.path as osp
import os

graph = torch.load(osp.join('../input/', 'RGG_100nodes.pt'))

gS = SAGEConv(graph.num_node_features, 2, root_weight=False, bias=False)

GAESage = GAE(gS)

GAESage.load_state_dict(torch.load( '../models/GAESage'))