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
from .fragment_encoder import *
from .vnn.models.vn_layers import *
from .vnn.models.utils.vn_dgcnn_util import get_graph_feature
from .egnn_vn_point_cloud import *
    

class Encoder_point_cloud(nn.Module):
    def __init__(self, input_nf = 45, edges_in_d = 5, n_knn = 10, conv_dims = [64, 64, 128, 256], num_components = 64, fragment_library_dim = 64, N_fragment_layers = 2, append_noise = False, N_members = 72, EGNN_layer_dim = 64, N_EGNN_layers = 3, pooling_MLP = True, shared_encoders = False, subtract_latent_space = False, variational = False, variational_mode = 'both', variational_GNN = False, variational_GNN_mol = False, mix_node_inv_to_equi = False, mix_shape_to_nodes = False, ablate_HvarCat = False, ablateEqui = False, old_EGNN = False):
        super(Encoder_point_cloud, self).__init__()

        self.input_nf = input_nf
        self.edges_in_d = edges_in_d

        self.conv_dims = conv_dims
        self.num_components = num_components
        self.fragment_library_dim = fragment_library_dim
        self.EGNN_layer_dim = EGNN_layer_dim
        self.N_EGNN_layers = N_EGNN_layers
        self.n_knn = n_knn
        self.pooling_MLP = pooling_MLP
        self.shared_encoders = shared_encoders
        self.subtract_latent_space = subtract_latent_space
        self.N_fragment_layers = N_fragment_layers
        self.N_members = N_members
        self.append_noise = append_noise
        self.variational = variational
        self.variational_mode = variational_mode
        self.variational_GNN = variational_GNN
        self.variational_GNN_mol = variational_GNN_mol
        self.mix_node_inv_to_equi = mix_node_inv_to_equi
        self.mix_shape_to_nodes = mix_shape_to_nodes
        self.ablate_HvarCat = ablate_HvarCat
        
        self.ablateEqui = ablateEqui

        self.fragment_encoder = FragmentLibraryEncoder(
            input_nf = input_nf, 
            edges_in_d = edges_in_d, 
            output_dim = fragment_library_dim, 
            N_layers = N_fragment_layers, 
            append_noise = append_noise, 
            N_members = N_members,
            
            old_EGNN = old_EGNN,
        )
        
        self.GraphEncoder = EGNN_VN_Encoder_point_cloud(
            node_input_dim = input_nf + fragment_library_dim, 
            edges_in_d = edges_in_d, 
            num_components = num_components, 
            EGNN_layer_dim = EGNN_layer_dim, 
            n_knn = n_knn, 
            conv_dims = conv_dims, 
            pooling_MLP = pooling_MLP,
            N_EGNN_layers = N_EGNN_layers,
            variational_GNN = self.variational_GNN,
            variational_GNN_mol = self.variational_GNN_mol,
            mix_node_inv_to_equi = self.mix_node_inv_to_equi,
            mix_shape_to_nodes = self.mix_shape_to_nodes,
            ablate_HvarCat = self.ablate_HvarCat,
            
            old_EGNN = old_EGNN,
        )
        
        if variational:
            if (variational_mode == 'both') | (variational_mode == 'equi'):
                self.VariationalEncoder_equi = nn.Sequential(
                    VNLinearAndLeakyReLU(num_components, num_components, use_batchnorm=False, negative_slope=0.2),
                    VNLinear(num_components, 2*num_components),
                )
                self.VariationalEncoder_equi_T = VNStdFeature(num_components, dim=3, normalize_frame=False)
                self.VariationEncoder_equi_linear = nn.Sequential(nn.Linear(num_components*3, num_components),
                        nn.LeakyReLU(negative_slope=0.2),
                        nn.Linear(num_components, num_components),
                        )

            if (variational_mode == 'both') | (variational_mode == 'inv'):
                self.VariationalEncoder_inv = nn.Sequential(
                    nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(EGNN_layer_dim, 2*EGNN_layer_dim),
                )

        if not self.shared_encoders:
            self.SubGraphEncoder = EGNN_VN_Encoder_point_cloud(
                node_input_dim = input_nf + fragment_library_dim, 
                edges_in_d = edges_in_d, 
                num_components = num_components, 
                EGNN_layer_dim = EGNN_layer_dim, 
                n_knn = n_knn, 
                conv_dims = conv_dims, 
                pooling_MLP = pooling_MLP,
                N_EGNN_layers = N_EGNN_layers,
                variational_GNN = False,
                variational_GNN_mol = False, 
                mix_node_inv_to_equi = self.mix_node_inv_to_equi,
                mix_shape_to_nodes = self.mix_shape_to_nodes,
                ablate_HvarCat = self.ablate_HvarCat,
                
                old_EGNN = old_EGNN,
            )

        self.Equi_linear_leaky_1 = VNLinearAndLeakyReLU(num_components*3 + int(self.subtract_latent_space)*num_components, num_components * 2, use_batchnorm=False, negative_slope=0.2)
        self.Equi_linear_leaky_2 = VNLinearAndLeakyReLU(num_components * 2, num_components, use_batchnorm=False, negative_slope=0.2)
        self.Equi_linear_leaky_3 = VNLinearAndLeakyReLU(num_components, num_components, use_batchnorm=False, negative_slope=0.2)
        self.T_layer = VNLinearAndLeakyReLU(num_components, 3, use_batchnorm=False, negative_slope=0.2)


    def encode_fragment_library(self, fragment_batch, device = torch.device('cpu')):

        fragment_library_features, fragment_library_node_features, fragment_library_batch = self.fragment_encoder(
            fragment_batch.x, 
            fragment_batch.edge_index, 
            fragment_batch.pos, 
            fragment_batch.edge_attr, 
            fragment_batch.batch,
            device = device,
        )

        return fragment_library_features, fragment_library_node_features, fragment_library_batch


    def encode(self, x, edge_index, pos, points, points_atom_index, edge_attr, batch_size, select_indices, select_indices_batch, shared_encoders = True, device = torch.device('cpu'), use_variational_GNN = False, variational_GNN_factor = 1.0, interpolate_to_GNN_prior = 0.0, h_interpolate = None):
                
        if shared_encoders:
            x_inv, Z_equi, Z_inv, Z_equi_select, Z_inv_select, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped = self.GraphEncoder(
                x, 
                edge_index, 
                pos,
                edge_attr,
                batch_size,
                points, 
                points_atom_index,
                select_indices = select_indices, 
                select_indices_batch = select_indices_batch,
                device = device,
                use_variational_GNN = use_variational_GNN,
                variational_GNN_factor = variational_GNN_factor, 
                interpolate_to_GNN_prior = interpolate_to_GNN_prior,
                h_interpolate = h_interpolate,
            )
        
        else:
            x_inv, Z_equi, Z_inv, Z_equi_select, Z_inv_select, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped = self.SubGraphEncoder(
                x, 
                edge_index, 
                pos,
                edge_attr,
                batch_size,
                points, 
                points_atom_index,
                select_indices = select_indices, 
                select_indices_batch = select_indices_batch,
                device = device,
                use_variational_GNN = use_variational_GNN,
                variational_GNN_factor = variational_GNN_factor, 
                interpolate_to_GNN_prior = interpolate_to_GNN_prior,
            )

        return x_inv, Z_equi, Z_inv, Z_equi_select, Z_inv_select, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped


    def mix_codes(self, batch_size, Z_equi, Z_inv, Z_equi_subgraph, Z_inv_subgraph, Z_equi_select, Z_inv_select):
        
        if self.ablateEqui:
            Z_equi = torch.zeros_like(Z_equi) # only ablating the equivariant information from the encoded molecule.
        
        if self.subtract_latent_space:
            Z_equivariant = torch.cat((Z_equi, Z_equi_subgraph, Z_equi_select, Z_equi - Z_equi_subgraph), dim = 1)
        else:
            Z_equivariant = torch.cat((Z_equi, Z_equi_subgraph, Z_equi_select), dim = 1)
        
        if self.subtract_latent_space:
            Z_invariant = torch.cat((Z_inv, Z_inv_subgraph, Z_inv_select, Z_inv - Z_inv_subgraph), dim = 1) 
        else:
            Z_invariant = torch.cat((Z_inv, Z_inv_subgraph, Z_inv_select), dim = 1)
        
        Z_equivariant = self.Equi_linear_leaky_1(Z_equivariant)
        Z_equivariant = self.Equi_linear_leaky_2(Z_equivariant)
        Z_equivariant = self.Equi_linear_leaky_3(Z_equivariant)
        
        T_equivariant = self.T_layer(Z_equivariant) 
        Z_T_invariant = torch.einsum('bij,bjk->bik', Z_equivariant, T_equivariant.permute(0,2,1))

        Z_T_invariant = Z_T_invariant.reshape(batch_size, -1)
        
        Z = torch.cat((Z_invariant, Z_T_invariant), dim = 1)

        return Z


    def forward(self, batch_size, x, edge_index, edge_attr, pos, points, points_atom_index, x_library_fragment_index, x_subgraph, subgraph_edge_index, subgraph_edge_attr, subgraph_pos, subgraph_points, subgraph_points_atom_index, x_subgraph_library_fragment_index, query_indices, query_indices_batch, fragment_batch, device = torch.device('cpu')):

        #---------------------------------------------------
        # Encoding
        #---------------------------------------------------

        fragment_library_features, fragment_library_node_features, fragment_library_batch = self.encode_fragment_library(fragment_batch, device = device)

        x = torch.cat((x, fragment_library_features[x_library_fragment_index]), dim = 1)
        
        x_subgraph = torch.cat((x_subgraph, fragment_library_features[x_subgraph_library_fragment_index]), dim = 1)
        
        _, Z_equi, Z_inv, _, _, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped = self.encode(
            x,
            edge_index, 
            pos, 
            points, 
            points_atom_index, 
            edge_attr, 
            batch_size,
            select_indices = None,
            select_indices_batch = None,
            shared_encoders = True,
            device = device,
            use_variational_GNN = (self.variational_GNN) | (self.variational_GNN_mol),
            variational_GNN_factor = 1.0, 
            interpolate_to_GNN_prior = 0.0,
        )
        
        
        if self.variational:
            if (self.variational_mode == 'both') | (self.variational_mode == 'equi'):
                Z_equi = self.VariationalEncoder_equi(Z_equi) # equivariant, shape [B,C*2,3]
                Z_equi_mean, Z_equi_logvar = Z_equi.chunk(2, dim = 1) # equivariant, shape [B,C,3]
                Z_equi_logvar, _ = self.VariationalEncoder_equi_T(Z_equi_logvar) # invariant, shape [B, C, 3]
                Z_equi_logvar = Z_equi_logvar.reshape(batch_size, -1) # flattened to shape [B, C*3]
                Z_equi_logvar = self.VariationEncoder_equi_linear(Z_equi_logvar).unsqueeze(2).expand((-1,-1,3)) # invariant, shape [B, C, 3]
                Z_equi_std = torch.exp(0.5 * Z_equi_logvar) # invariant, shape [B, C, 1]
                Z_equi_eps = torch.randn_like(Z_equi_mean) # normal noise with shape [B,C,3]
                Z_equi = Z_equi_mean + Z_equi_std * Z_equi_eps # equivariant mean + isotropic noise (equivariant)
            else:
                Z_equi_mean = None
                Z_equi_std = None

            if (self.variational_mode == 'both') | (self.variational_mode == 'inv'):
                Z_inv = self.VariationalEncoder_inv(Z_inv)
                Z_inv_mean, Z_inv_logvar = Z_inv.chunk(2, dim = 1)
                Z_inv_std = torch.exp(0.5 * Z_inv_logvar)
                Z_inv_eps = torch.randn_like(Z_inv_mean)
                Z_inv = Z_inv_mean + Z_inv_std * Z_inv_eps
            else:
                Z_inv_mean = None
                Z_inv_std = None
        else:
            Z_equi_mean = None
            Z_equi_std = None
            Z_inv_mean = None
            Z_inv_std = None
            
                
        x_inv_subgraph, Z_equi_subgraph, Z_inv_subgraph, Z_equi_select, Z_inv_select, _, _, _, _, _ = self.encode(
            x_subgraph,
            subgraph_edge_index, 
            subgraph_pos, 
            subgraph_points, 
            subgraph_points_atom_index, 
            subgraph_edge_attr, 
            batch_size,
            select_indices = query_indices,
            select_indices_batch = query_indices_batch,
            shared_encoders = self.shared_encoders,
            device = device,
            use_variational_GNN = False,
            variational_GNN_factor = 1.0, 
            interpolate_to_GNN_prior = 0.0,
        )
        

        if self.ablateEqui:
            Z_equi = torch.zeros_like(Z_equi) # only ablating the equivariant information from the encoded molecule.
        

        graph_subgraph_select_features_concat = self.mix_codes(batch_size, Z_equi, Z_inv, Z_equi_subgraph, Z_inv_subgraph, Z_equi_select, Z_inv_select)

        h_subgraph = x_inv_subgraph.permute(0,2,1).reshape(-1, x_inv_subgraph.shape[1])
        h_select = h_subgraph[query_indices]

        return graph_subgraph_select_features_concat, h_subgraph, h_select, fragment_library_features, fragment_library_node_features, fragment_library_batch, Z_equi_mean, Z_equi_std, Z_inv_mean, Z_inv_std, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped

