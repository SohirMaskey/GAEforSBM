"""
Creates a graph by realizing an SBM on [0,1]
"""
import numpy as np
import os
import networkx as nx
import torch
from torch_geometric.utils import from_networkx    
from torch_geometric.utils import degree


N = 1000 #node size

samples = np.random.uniform(0,1,N)

#communities of the SBM
intervals = [1/4, 2/4, 3/4, 1] 

sizes = [] 

#calculates the amount of nodes in each community
for i in range(0, len(intervals)): 
    size = 0
    for j in range(0, len(samples)):
        if i == 0:
            if samples[j] > 0 and samples[j] <= intervals[0]:
                size = size + 1
        elif samples[j] > intervals[i-1] and samples[j] <= intervals[i]:
            size = size+1
    sizes.append(size)
    
samples = torch.tensor(samples).reshape(N,1)

#Choose some probabilities for each community
probsPerInt = [6/16, 1/16, 2/16, 3/16] 

probs = []

#correct form for module from networkx
for i in range(0, len(probsPerInt)):
    prob = [1/16] * len(probsPerInt)
    prob[i] = probsPerInt[i]
    probs.append(prob)
    
#realization of the SBM defined by above parameters    
g = nx.stochastic_block_model(sizes, probs, seed=0) 

#transform to pytorch Data object
graph = from_networkx(g) 

#calculate tensor with degree of each node
d = degree(graph.edge_index[0], N, dtype=torch.long).reshape(N,1) 

graph.x = d/N
#graph.x = graph.x.type(torch.LongTensor)
graph.x = graph.x.type(torch.FloatTensor)
graph.pos = samples.type(torch.FloatTensor)

torch.save(graph, 
                os.path.join('../input', 
                    f'RGG_1000nodes.pt'))