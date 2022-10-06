import torch_geometric
import torch
import torch_scatter
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import networkx as nx
import random

from .EGNN import *

class FragmentLibraryEncoder(nn.Module):
    def __init__(self, input_nf = 45, edges_in_d = 5, output_dim = 64, N_layers = 2, append_noise = False, N_members = 72, no_3D = False, old_EGNN = False):
        super(FragmentLibraryEncoder, self).__init__()
        
        self.no_3D = no_3D
        self.output_dim = output_dim
        self.N_layers = N_layers
        self.append_noise = append_noise
        self.N_members = N_members

        self.EGNN_layers = nn.ModuleList([
            EGNN_static(input_nf = input_nf, # input number of node features
                         output_nf = output_dim, # output number of node/edge features
                         hidden_nf = output_dim, 
                         edges_in_d = edges_in_d, # input number of edge features
                         residual = False,
                         no_3D = no_3D,
                        
                         old_EGNN = old_EGNN,
            )
        ])

        for layer in range(1, N_layers):
            self.EGNN_layers.append(
                EGNN_static(input_nf = output_dim, 
                                 output_nf = output_dim if ((layer < (N_layers - 1)) | (self.append_noise == False)) else output_dim // 2,
                                 hidden_nf = output_dim, 
                                 edges_in_d = output_dim,
                                 residual = True if ((layer < (N_layers - 1)) | (self.append_noise == False)) else False,
                                 no_3D = no_3D,
                                 
                                 old_EGNN = old_EGNN,
                )
            )

        if self.append_noise:
            self.noise_embedding = torch.nn.Embedding(N_members, output_dim - (output_dim // 2))
        
    def forward(self, x, edge_index, pos, edge_attr, batch_index, device = torch.device('cpu')):

        h, _, edge_feat = self.EGNN_layers[0](x, edge_index, pos, edge_attr=edge_attr, node_attr=None)
        for EGNN_layer in self.EGNN_layers[1:]:
            h, _, edge_feat = EGNN_layer(h, edge_index, pos, edge_attr=edge_feat, node_attr=None)
        
        graph_features = torch_scatter.scatter_add(h, batch_index, dim = 0)

        if self.append_noise:
            graph_features = torch.cat([graph_features, self.noise_embedding(torch.arange(0, self.N_members, device = device))], dim = 1)
            h = torch.cat([h, self.noise_embedding(batch_index)], dim = 1)
        
        return graph_features, h, batch_index
