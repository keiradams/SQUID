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
from .decoder import *
from .encoder import *


class ROCS_Model_Point_Cloud(nn.Module):
    def __init__(self, input_nf = 45, edges_in_d = 5, n_knn = 10, conv_dims = [64, 64, 128, 256], num_components = 64, fragment_library_dim = 64, N_fragment_layers = 2, append_noise = False, N_members = 72, EGNN_layer_dim = 64, N_EGNN_layers = 3, output_MLP_hidden_dim = 64, pooling_MLP = True, shared_encoders = False, subtract_latent_space = False, variational = False, variational_mode = 'both', variational_GNN = False, variational_GNN_mol = False, mix_node_inv_to_equi = False, mix_shape_to_nodes = False, ablate_HvarCat = False, ablateEqui = False, old_EGNN = False):
        super(ROCS_Model_Point_Cloud, self).__init__()

        self.input_nf = input_nf
        self.edges_in_d = edges_in_d

        self.conv_dims = conv_dims
        self.num_components = num_components
        self.fragment_library_dim = fragment_library_dim
        self.EGNN_layer_dim = EGNN_layer_dim
        self.N_EGNN_layers = N_EGNN_layers
        self.n_knn = n_knn
        self.output_MLP_hidden_dim = output_MLP_hidden_dim
        self.pooling_MLP = pooling_MLP
        self.shared_encoders = shared_encoders
        self.subtract_latent_space = subtract_latent_space
        self.N_fragment_layers = N_fragment_layers
        self.N_members = N_members
        self.append_noise = append_noise
        self.variational = variational
        self.variational_mode = variational_mode

        if not self.subtract_latent_space:
            graph_subgraph_focal_features_concat_dim = num_components*3 + EGNN_layer_dim*3 # *3 is from flattening the dimensions from the (now invariant) equivariant latent space. *3 is from concatenating the invariant latent spaces (encoded, partial, focal)
        else:
            graph_subgraph_focal_features_concat_dim = num_components*3 + EGNN_layer_dim*4 # *4 is from concatenating the invariant latent spaces (encoded, partial, focal, encoded - partial)

        
        self.Encoder = Encoder_point_cloud(
            input_nf = input_nf, 
            edges_in_d = edges_in_d, 
            n_knn = n_knn, 
            conv_dims = conv_dims, 
            num_components = num_components, 
            fragment_library_dim = fragment_library_dim, 
            N_fragment_layers = N_fragment_layers, 
            append_noise = append_noise, 
            N_members = N_members, 
            EGNN_layer_dim = EGNN_layer_dim, 
            N_EGNN_layers = N_EGNN_layers, 
            pooling_MLP = pooling_MLP, 
            shared_encoders = shared_encoders, 
            subtract_latent_space = subtract_latent_space,
            variational = variational,
            variational_mode = variational_mode,
            variational_GNN = variational_GNN,
            variational_GNN_mol = variational_GNN_mol,
            mix_node_inv_to_equi = mix_node_inv_to_equi,
            mix_shape_to_nodes = mix_shape_to_nodes,
            ablate_HvarCat = ablate_HvarCat,
            
            ablateEqui = ablateEqui,
            
            old_EGNN = old_EGNN,
        )

        self.ROCS_scorer = EGNN_MLP(graph_subgraph_focal_features_concat_dim, 1, output_MLP_hidden_dim)


    def forward(self, batch_size, x, edge_index, edge_attr, pos, points, points_atom_index, x_library_fragment_index, x_subgraph, subgraph_edge_index, subgraph_edge_attr, subgraph_pos, subgraph_points, subgraph_points_atom_index, x_subgraph_library_fragment_index, query_indices, query_indices_batch, fragment_batch, device = torch.device('cpu')):
        # Encoding
        graph_subgraph_select_features_concat, h_subgraph, h_select, _, _, _, Z_equi_mean, Z_equi_std, Z_inv_mean, Z_inv_std, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped = self.Encoder(
            batch_size, 
            x, 
            edge_index, 
            edge_attr, 
            pos, 
            points, 
            points_atom_index, 
            x_library_fragment_index, 
            x_subgraph, 
            subgraph_edge_index, 
            subgraph_edge_attr, 
            subgraph_pos, 
            subgraph_points, 
            subgraph_points_atom_index, 
            x_subgraph_library_fragment_index, 
            query_indices, 
            query_indices_batch, 
            fragment_batch, 
            device = device, 
        )

        scores = self.ROCS_scorer(graph_subgraph_select_features_concat) # NOT sigmoided yet.

        return scores, Z_equi_mean, Z_equi_std, Z_inv_mean, Z_inv_std, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped



class Model_Point_Cloud_Switched(nn.Module):
    def __init__(self, input_nf = 45, edges_in_d = 5, n_knn = 10, conv_dims = [64, 64, 128, 256], num_components = 64, fragment_library_dim = 64, N_fragment_layers = 2, append_noise = False, N_members = 72, EGNN_layer_dim = 64, N_EGNN_layers = 3, output_MLP_hidden_dim = 64, pooling_MLP = True, shared_encoders = False, subtract_latent_space = False, variational = False, variational_mode = 'both', variational_GNN = False, variational_GNN_mol = False, mix_node_inv_to_equi = False, mix_shape_to_nodes = False, ablate_HvarCat = False, predict_pairwise_properties = False, predict_mol_property = False, ablateEqui = False, old_EGNN = False):
        super(Model_Point_Cloud_Switched, self).__init__()

        self.input_nf = input_nf
        self.edges_in_d = edges_in_d

        self.conv_dims = conv_dims
        self.num_components = num_components
        self.fragment_library_dim = fragment_library_dim
        self.EGNN_layer_dim = EGNN_layer_dim
        self.N_EGNN_layers = N_EGNN_layers
        self.n_knn = n_knn
        self.output_MLP_hidden_dim = output_MLP_hidden_dim
        self.pooling_MLP = pooling_MLP
        self.shared_encoders = shared_encoders
        self.subtract_latent_space = subtract_latent_space
        self.N_fragment_layers = N_fragment_layers
        self.N_members = N_members
        self.append_noise = append_noise
        self.variational = variational
        self.variational_mode = variational_mode

        if not self.subtract_latent_space:
            graph_subgraph_focal_features_concat_dim = num_components*3 + EGNN_layer_dim*3 # *3 is from flattening the dimensions from the (now invariant) equivariant latent space. *3 is from concatenating the invariant latent spaces (encoded, partial, focal)
        else:
            graph_subgraph_focal_features_concat_dim = num_components*3 + EGNN_layer_dim*4 # *4 is from concatenating the invariant latent spaces (encoded, partial, focal, encoded - partial)

        
        self.Encoder = Encoder_point_cloud(
            input_nf = input_nf, 
            edges_in_d = edges_in_d, 
            n_knn = n_knn, 
            conv_dims = conv_dims, 
            num_components = num_components, 
            fragment_library_dim = fragment_library_dim, 
            N_fragment_layers = N_fragment_layers, 
            append_noise = append_noise, 
            N_members = N_members, 
            EGNN_layer_dim = EGNN_layer_dim, 
            N_EGNN_layers = N_EGNN_layers, 
            pooling_MLP = pooling_MLP, 
            shared_encoders = shared_encoders, 
            subtract_latent_space = subtract_latent_space,
            variational = variational,
            variational_mode = variational_mode,
            variational_GNN = variational_GNN,
            variational_GNN_mol = variational_GNN_mol,
            mix_node_inv_to_equi = mix_node_inv_to_equi,
            mix_shape_to_nodes = mix_shape_to_nodes,
            ablate_HvarCat = ablate_HvarCat,
            
            ablateEqui = ablateEqui,
            
            old_EGNN = old_EGNN
        )

        self.Decoder = GraphDecoderSwitched(
            Z_dim = graph_subgraph_focal_features_concat_dim, 
            fragment_library_dim = fragment_library_dim, 
            EGNN_layer_dim = EGNN_layer_dim, 
            output_MLP_hidden_dim = output_MLP_hidden_dim,
            )
        
        
        self.predict_pairwise_properties = predict_pairwise_properties
        if self.predict_pairwise_properties:
            self.PairwiseMixing = nn.Sequential(
                nn.Linear(EGNN_layer_dim*2, EGNN_layer_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim),
            )
            self.PairwisePropertyPredictor = nn.Sequential(
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim//2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim//2, EGNN_layer_dim//4),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim//4, 1),
            )
        
        self.predict_mol_property = predict_mol_property
        if self.predict_mol_property:
            self.MolPropertyPredictor = nn.Sequential(
                nn.Linear(EGNN_layer_dim, EGNN_layer_dim//2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim//2, EGNN_layer_dim//4),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(EGNN_layer_dim//4, 1),
            )
        
    def pairwise_predict(self, h_reshaped, pairwise_indices_1_select, pairwise_indices_2_select):
        h_aggregated = h_reshaped.mean(-1) # (B)x(F); mean aggregation so we can reasonably compare molecules of different sizes.
        h_pairwise_cat_ab = torch.cat((h_aggregated[pairwise_indices_1_select], h_aggregated[pairwise_indices_2_select]), dim = 1)
        h_pairwise_cat_ba = torch.cat((h_aggregated[pairwise_indices_2_select], h_aggregated[pairwise_indices_1_select]), dim = 1)
        h_pairwise_mixed = self.PairwiseMixing(h_pairwise_cat_ab) + self.PairwiseMixing(h_pairwise_cat_ba) # symmetry preserving
        pairwise_properties_out = self.PairwisePropertyPredictor(h_pairwise_mixed)
        return pairwise_properties_out
    
    def mol_property_predict(self, h_reshaped):
        h_aggregated = h_reshaped.sum(-1)
        mol_property_out = self.MolPropertyPredictor(h_aggregated)
        return mol_property_out


    def forward(self, batch_size, x, edge_index, edge_attr, pos, points, points_atom_index, x_library_fragment_index, x_subgraph, subgraph_edge_index, subgraph_edge_attr, subgraph_pos, subgraph_points, subgraph_points_atom_index, x_subgraph_library_fragment_index, query_indices, query_indices_batch, fragment_batch, next_atom_type_library_idx, stop_mask, stop_focal_mask, masked_focal_batch_index_reindexed, focal_attachment_index_rel_partial, next_atom_attachment_indices, masked_next_atom_attachment_batch_index_reindexed, masked_multihot_next_attachments, all_stop = False, pairwise_indices_1_select = None, pairwise_indices_2_select = None, device = torch.device('cpu')):
        
        # Encoding
        graph_subgraph_select_features_concat, h_subgraph, h_select, fragment_library_features, fragment_library_node_features, fragment_library_batch, Z_equi_mean, Z_equi_std, Z_inv_mean, Z_inv_std, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped = self.Encoder(
            batch_size, 
            x, 
            edge_index, 
            edge_attr, 
            pos, 
            points, 
            points_atom_index, 
            x_library_fragment_index, 
            x_subgraph, 
            subgraph_edge_index, 
            subgraph_edge_attr, 
            subgraph_pos, 
            subgraph_points, 
            subgraph_points_atom_index, 
            x_subgraph_library_fragment_index, 
            query_indices, 
            query_indices_batch, 
            fragment_batch,
            device = device,
        )
        
        # predicting any properties from latent space
        # h_reshaped is of shape (B)x(F)x(N)
        if self.predict_pairwise_properties:
            pairwise_properties_out = self.pairwise_predict(h_reshaped, pairwise_indices_1_select, pairwise_indices_2_select)
        else:
            pairwise_properties_out = None
        
        if self.predict_mol_property:
            mol_property_out = self.mol_property_predict(h_reshaped)
        else:
            mol_property_out = None
            

        # Decoding
        stop_logits, next_atom_fragment_logits, fragment_attachment_scores_softmax, next_fragment_attachment_scores_softmax, bond_types_logits = self.Decoder(
            fragment_library_features, 
            fragment_library_node_features, 
            graph_subgraph_select_features_concat, 
            h_subgraph, 
            h_select, 
            next_atom_type_library_idx, 
            stop_mask, 
            stop_focal_mask, 
            masked_focal_batch_index_reindexed, 
            focal_attachment_index_rel_partial, 
            next_atom_attachment_indices, 
            masked_next_atom_attachment_batch_index_reindexed, 
            masked_multihot_next_attachments, 
            all_stop = all_stop,
        )
        
        return stop_logits, next_atom_fragment_logits, fragment_attachment_scores_softmax, next_fragment_attachment_scores_softmax, bond_types_logits, Z_equi_mean, Z_equi_std, Z_inv_mean, Z_inv_std, h_mean, h_std, h_reshaped_gnn, h_predicted_reshaped, h_reshaped, pairwise_properties_out, mol_property_out

