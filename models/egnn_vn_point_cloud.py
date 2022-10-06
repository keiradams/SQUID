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
from .vnn.models.vn_layers import *
from .vnn.models.utils.vn_dgcnn_util import get_graph_feature


class EGNN_VN_Encoder_point_cloud(nn.Module):
    def __init__(self, node_input_dim = (45 + 64), edges_in_d = 5, num_components = 64, EGNN_layer_dim = 64, n_knn = 10, conv_dims = [64, 64, 128, 256], pooling_MLP = True, N_EGNN_layers = 3, variational_GNN = False, variational_GNN_mol = False, mix_node_inv_to_equi = False, mix_shape_to_nodes = False, ablate_HvarCat = False, old_EGNN = False):
        super(EGNN_VN_Encoder_point_cloud, self).__init__()
        
        self.N_EGNN_layers = N_EGNN_layers
        self.pooling = 'mean'
        self.n_knn = n_knn
        self.num_components = num_components
        self.pooling_MLP = pooling_MLP
        self.conv_dims = conv_dims

        self.EGNN_layer_dim = EGNN_layer_dim
        self.point_invariant_mlp_hidden_dim = EGNN_layer_dim
        self.h_dim = EGNN_layer_dim
        self.pooling_MLP_dim = EGNN_layer_dim
        
        self.ablate_HvarCat = ablate_HvarCat
        self.mix_shape_to_nodes = mix_shape_to_nodes
        if self.mix_shape_to_nodes:
            self.std_feature_mix_shape_to_nodes = VNStdFeature(self.num_components*2, dim=4, normalize_frame=False)
            
        
        self.EGNN_layers = nn.ModuleList([
            EGNN_static(
                input_nf = node_input_dim + int(self.mix_shape_to_nodes) * self.num_components*2*3,  # input number of node features
                output_nf = EGNN_layer_dim, # output number of node/edge features
                hidden_nf = EGNN_layer_dim + int(self.mix_shape_to_nodes) * EGNN_layer_dim,  
                edges_in_d = edges_in_d, # input number of edge features
                residual = False,
                
                old_EGNN = old_EGNN,
            )
        ])

        for layer in range(1, N_EGNN_layers):
            self.EGNN_layers.append(
                EGNN_static(
                    input_nf = EGNN_layer_dim,
                    output_nf = EGNN_layer_dim,
                    hidden_nf = EGNN_layer_dim, 
                    edges_in_d = EGNN_layer_dim,
                    residual = True,
                    
                    old_EGNN = old_EGNN,
                )
            )
        
        self.variational_GNN = variational_GNN
        if self.variational_GNN:
            self.variational_GNN_encoder = nn.Sequential(
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim * 2),
            )
        
        self.variational_GNN_mol = variational_GNN_mol
        if self.variational_GNN_mol:
            self.variational_GNN_mol_encoder = nn.Sequential(
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim * 2),
            )
            self.h_predictor = nn.Sequential(
                nn.Linear(self.num_components*2*3 * 2 + EGNN_layer_dim, EGNN_layer_dim*4),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim*4, EGNN_layer_dim*2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim*2, EGNN_layer_dim),
            )
        
        self.mix_node_inv_to_equi = mix_node_inv_to_equi
        if self.mix_node_inv_to_equi:
            self.project_h_embeddings = nn.Sequential(
                nn.Linear(EGNN_layer_dim, num_components),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(num_components, num_components * num_components // 2),
            )
            self.Equi_linear_leaky_mixing = VNLinearAndLeakyReLU(num_components + num_components // 2, num_components, use_batchnorm=False, negative_slope=0.2)
        

        self.conv1 = VNLinearLeakyReLU(2, self.conv_dims[0]//3) # VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(self.conv_dims[0]//3*2, self.conv_dims[1]//3)
        self.conv3 = VNLinearLeakyReLU(self.conv_dims[1]//3*2, self.conv_dims[2]//3)
        self.conv4 = VNLinearLeakyReLU(self.conv_dims[2]//3*2, self.conv_dims[3]//3)
        self.conv5 = VNLinearLeakyReLU(self.conv_dims[3]//3 + self.conv_dims[2]//3 + self.conv_dims[1]//3 + self.conv_dims[0]//3, self.num_components, dim=4, share_nonlinearity=True)

        if self.pooling == 'max':
            self.pool1 = VNMaxPool(self.conv_dims[0]//3)
            self.pool2 = VNMaxPool(self.conv_dims[1]//3)
            self.pool3 = VNMaxPool(self.conv_dims[2]//3)
            self.pool4 = VNMaxPool(self.conv_dims[3]//3)
        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
            
        self.std_feature = VNStdFeature(self.num_components*2, dim=4, normalize_frame=False)
        
        self.point_invariant_mlp = nn.Sequential(nn.Linear(self.num_components*2*3 + self.h_dim, self.point_invariant_mlp_hidden_dim * 2),
                        nn.LeakyReLU(negative_slope=0.2),
                        nn.Linear(self.point_invariant_mlp_hidden_dim * 2, self.point_invariant_mlp_hidden_dim),
                        nn.LeakyReLU(negative_slope=0.2),
                        nn.Linear(self.point_invariant_mlp_hidden_dim, self.point_invariant_mlp_hidden_dim),
                        )
        
        
        if self.pooling_MLP:
            self.mlp = nn.Sequential(nn.Linear(self.point_invariant_mlp_hidden_dim, self.pooling_MLP_dim),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(self.pooling_MLP_dim, self.pooling_MLP_dim),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(self.pooling_MLP_dim, self.pooling_MLP_dim),
                                    )

        
    def forward(self, h, edge_index, pos, edge_attr, batch_size, points, points_atom_index, select_indices = None, select_indices_batch = None, device = torch.device('cpu'), use_variational_GNN = False, variational_GNN_factor = 1.0, interpolate_to_GNN_prior = 0.0, h_interpolate = None):
        
        
        pos_reshaped = (pos.reshape(batch_size, -1, 3)).permute(0,2,1)
        points_reshaped = (points.reshape(batch_size, -1, 3)).permute(0,2,1) # (B)x(3)x(N_points_per_cloud)
        points_atom_index_reshape = points_atom_index.reshape(batch_size, -1) 

        batch_size = points_reshaped.size(0)

        x = points_reshaped.unsqueeze(1) # (B)x(1)x(3)x(N_points_per_cloud)
        x = get_graph_feature(x, k=self.n_knn, device = device)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn, device = device)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn, device = device)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn, device = device)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x) # x is now shape [batch, num_components, 3, N_points_per_cloud]

        num_points = x.size(-1)

        x_atom_pooled = torch_scatter.scatter_mean(x, points_atom_index_reshape.unsqueeze(1).unsqueeze(1)) # (B)x(C)x(3)x(N_nodes)
        
        
        if self.mix_shape_to_nodes:
            
            num_nodes = x_atom_pooled.size(-1)
            
            x_mean_expanded = x_atom_pooled.sum(dim=-1, keepdim=True).expand(x_atom_pooled.size())
            x_gnn_invariant, _ = self.std_feature_mix_shape_to_nodes(torch.cat((x_atom_pooled, x_mean_expanded), dim = 1)) # shape (B, dim, 3, N_nodes)
            x_gnn_invariant_reshaped = x_gnn_invariant.reshape(batch_size, -1, num_nodes) # rotation-invariant features per node, B x F x N_nodes
            x_gnn_to_cat_to_h = x_gnn_invariant_reshaped.permute(0,2,1).reshape(batch_size * num_nodes, -1) # B x F x N_nodes --(permute)--> B x N_nodes x F --(reshape)--> B*N_nodes x F
            
            h = torch.cat((h, x_gnn_to_cat_to_h), dim = 1) # B*N_nodes x F
            
        
        h, _, edge_feat = self.EGNN_layers[0](h, edge_index, pos, edge_attr=edge_attr, node_attr=None)
        for EGNN_layer in self.EGNN_layers[1:]:
            h, _, edge_feat = EGNN_layer(h, edge_index, pos, edge_attr=edge_feat, node_attr=None)
        
        # try variationally encoding h embeddings here...
        if (self.variational_GNN) & (use_variational_GNN) & (self.variational_GNN_mol == False):
            h_variational = self.variational_GNN_encoder(h) # atom-wise variational encoding
            h_mean, h_logvar = h_variational.chunk(2, dim = 1) 
            h_std = torch.exp(0.5 * h_logvar)
            h_eps = torch.randn_like(h_mean) * variational_GNN_factor
            
            if interpolate_to_GNN_prior > 1e-4: #e.g., > 0.0
                h_mean = torch.lerp(h_mean, torch.zeros_like(h_mean), interpolate_to_GNN_prior)
                h_std = torch.lerp(h_std, torch.ones_like(h_std), interpolate_to_GNN_prior)
            
            h = h_mean + h_std * h_eps
        else:
            h_mean = None
            h_std = None
        
        if h_interpolate is not None:
            h = h_interpolate
        
        h_reshaped = h.unsqueeze(0).reshape(batch_size, -1, h.shape[1]).permute(0,2,1) # (B*N)x(F) -> (B)x(N)x(F) -> (B)x(F)x(N)
        
        
        
        # mixing invariant GNN node embeddings into VN equivariant node embeddings
        if self.mix_node_inv_to_equi:            
            h_projected = self.project_h_embeddings(h_reshaped.permute(0,2,1)) # (B) x (N) x (C' * C)
            h_projected_reshaped = h_projected.reshape(h_projected.shape[0], h_projected.shape[1], self.num_components // 2, self.num_components) # (B) x (N) x (C') x (C)
            x_atom_pooled_mixed = torch.einsum('bijk,bikm->bijm', h_projected_reshaped, x_atom_pooled.permute(0,3,1,2)) # (B) x (N) x (C') x (3)
            x_atom_pooled = self.Equi_linear_leaky_mixing(torch.cat((x_atom_pooled.permute(0,3,1,2), x_atom_pooled_mixed), dim = 2).permute(0,2,3,1)) # (B)x(C)x(3)x(N_nodes)
                        
        
        
        num_nodes = x_atom_pooled.size(-1)

        Z_equivariant = x_atom_pooled.sum(dim=-1, keepdim=False)
        
        # pooling over select points in the cloud
        if select_indices != None:
            Z_equivariant_select = torch_scatter.scatter_add(x_atom_pooled.permute(0,3,1,2).reshape(-1, self.num_components, 3)[select_indices], select_indices_batch, dim = 0)
            
        x_mean_expanded = x_atom_pooled.sum(dim=-1, keepdim=True).expand(x_atom_pooled.size())
        x_cat_x_mean = torch.cat((x_atom_pooled, x_mean_expanded), 1)
        x_invariant, trans = self.std_feature(x_cat_x_mean) # x_invariant now has shape (B, dim, 3, N_nodes)
        
        x_invariant = x_invariant.reshape(batch_size, -1, num_nodes) # rotation-invariant features per node, B x F x N_nodes

        #!#
        if (self.variational_GNN_mol) & (use_variational_GNN) & (self.variational_GNN == False):
            h_mol = h_reshaped.sum(-1) # (B)x(F)x(N) -> (B)x(F)
        
            h_mol_mean_logvar = self.variational_GNN_mol_encoder(h_mol)
            h_mol_mean, h_mol_logvar = h_mol_mean_logvar.chunk(2, dim = 1)
            h_mol_std = torch.exp(0.5 * h_mol_logvar)
            h_mol_eps = torch.randn_like(h_mol_mean) * variational_GNN_factor # use the same variational_GNN_factor
            
            if interpolate_to_GNN_prior > 1e-4: #e.g., > 0.0 # use the same interpolate_to_GNN_prior
                h_mol_mean = torch.lerp(h_mol_mean, torch.zeros_like(h_mol_mean), interpolate_to_GNN_prior)
                h_mol_std = torch.lerp(h_mol_std, torch.ones_like(h_mol_std), interpolate_to_GNN_prior)
            
            h_mol = h_mol_mean + h_mol_std * h_mol_eps
            
            x_global = x_invariant.sum(-1) # (B)x(F)x(N) -> (B)x(F)
            x_global_h_mol = torch.cat((x_global, h_mol), dim = 1) # (B)x(F)
            x_invariant_x_global_h_mol_cat = torch.cat((x_invariant, x_global_h_mol.unsqueeze(2).expand(-1, -1, x_invariant.shape[2])), dim = 1) # (B)x(F)x(N)
            h_reshaped_gnn = h_reshaped
            h_predicted_reshaped = self.h_predictor(x_invariant_x_global_h_mol_cat.permute(0,2,1)).permute(0,2,1) # (B)x(F)x(N)
            h_reshaped = h_predicted_reshaped
            
            h_mean = h_mol_mean 
            h_std = h_mol_std
            

        else:
            h_reshaped_gnn = None 
            h_predicted_reshaped = None
            
        
        if self.ablate_HvarCat:
            x_invariant = torch.cat((x_invariant, torch.zeros_like(h_reshaped)), dim = 1)
        else:
            x_invariant = torch.cat((x_invariant, h_reshaped), dim = 1) # concatenate h embeddings with x_invariant: (B)x(F)x(N)
        
        
        x_invariant = self.point_invariant_mlp(x_invariant.permute(0,2,1)).permute(0,2,1)
        Z_invariant = x_invariant.sum(dim=-1, keepdim=False)
        
        if self.pooling_MLP:
            Z_invariant = self.mlp(Z_invariant)
        
        # pooling over select points in the cloud
        if select_indices != None:
            Z_invariant_select = torch_scatter.scatter_add(x_invariant.permute(0,2,1).reshape(-1, x_invariant.shape[1])[select_indices], select_indices_batch, dim = 0)
            
            if self.pooling_MLP:
                Z_invariant_select = self.mlp(Z_invariant_select)
            
            return x_invariant, Z_equivariant, Z_invariant, Z_equivariant_select, Z_invariant_select, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped
        
        else:
            return x_invariant, Z_equivariant, Z_invariant, None, None, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped

