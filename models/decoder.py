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

class GraphDecoderSwitched(nn.Module):
    def __init__(self,  Z_dim = 64, fragment_library_dim = 64, EGNN_layer_dim = 64, output_MLP_hidden_dim = 64):
        super(GraphDecoderSwitched, self).__init__()

        self.Z_dim = Z_dim 
        self.fragment_library_dim = fragment_library_dim 
        self.EGNN_layer_dim = EGNN_layer_dim 
        self.output_MLP_hidden_dim = output_MLP_hidden_dim

        self.MLP_stop_scores = EGNN_MLP(Z_dim, 1, output_MLP_hidden_dim) 
        self.MLP_fragment_attachment_scores = EGNN_MLP(Z_dim + EGNN_layer_dim , 1, output_MLP_hidden_dim) 
        self.MLP_next_fragment_scores = EGNN_MLP(Z_dim + EGNN_layer_dim + fragment_library_dim, 1, output_MLP_hidden_dim)
        self.MLP_next_fragment_attachment_scores = EGNN_MLP(Z_dim  + EGNN_layer_dim + fragment_library_dim + fragment_library_dim, 1, output_MLP_hidden_dim)
        self.MLP_bond_types = EGNN_MLP(Z_dim + EGNN_layer_dim + fragment_library_dim + fragment_library_dim, 4, output_MLP_hidden_dim)       

    def forward(self, fragment_library_features, fragment_library_node_features, graph_subgraph_focal_features_concat, h_partial, h_focal, next_atom_type_library_idx, stop_mask, stop_focal_mask, masked_focal_batch_index_reindexed, focal_attachment_index_rel_partial, next_atom_attachment_indices, masked_next_atom_attachment_batch_index_reindexed, masked_multihot_next_attachments, all_stop = False):
        
        #---------------------------------------------------
        # Decoding
        #---------------------------------------------------

        # PREDICT whether to STOP
        stop_logits = self.decode_stop(graph_subgraph_focal_features_concat)

        if all_stop:
            return stop_logits, None, None, None, None

        # Predicting focal fragment attachment point
        h_focal_masked = h_focal[stop_focal_mask]
        graph_subgraph_focal_features_concat_masked = graph_subgraph_focal_features_concat[stop_mask] 
        graph_subgraph_focal_focalAttachments_features_concat_masked = torch.cat((graph_subgraph_focal_features_concat_masked[masked_focal_batch_index_reindexed], h_focal_masked), dim = 1)
        fragment_attachment_scores = self.MLP_fragment_attachment_scores(graph_subgraph_focal_focalAttachments_features_concat_masked)
        fragment_attachment_scores_softmax = torch_scatter.composite.scatter_softmax(fragment_attachment_scores.squeeze(1), masked_focal_batch_index_reindexed)


        # PREDICT next atom/fragment in classification setting
        focal_atom_features_masked = h_partial[focal_attachment_index_rel_partial[stop_mask]]
        graph_subgraph_focal_focalAtom_features_concat_masked = torch.cat((graph_subgraph_focal_features_concat_masked, focal_atom_features_masked), dim = 1)
        next_atom_fragment_logits = self.decode_next_atom_fragment(graph_subgraph_focal_focalAtom_features_concat_masked, fragment_library_features)

        nextAtomFragment_features_masked = fragment_library_features[next_atom_type_library_idx[stop_mask]] 
        graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat_masked = torch.cat((graph_subgraph_focal_focalAtom_features_concat_masked, nextAtomFragment_features_masked), dim = 1)


        # 3. Predict attachment point in the next atom/fragment
        next_fragment_attachment_features = fragment_library_node_features[next_atom_attachment_indices]
        graph_subgraph_focal_nextAtomFragment_focalAtom_nextFragmentAttachments_features_concat_masked = torch.cat((graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat_masked[masked_next_atom_attachment_batch_index_reindexed], next_fragment_attachment_features), dim = 1)
        next_fragment_attachment_scores = self.MLP_next_fragment_attachment_scores(graph_subgraph_focal_nextAtomFragment_focalAtom_nextFragmentAttachments_features_concat_masked)
        next_fragment_attachment_scores_softmax = torch_scatter.composite.scatter_softmax(next_fragment_attachment_scores.squeeze(1), masked_next_atom_attachment_batch_index_reindexed)
       
        # 4. Predict bond type
        nextAtomEquivalent_features = fragment_library_node_features[next_atom_attachment_indices[masked_multihot_next_attachments]]
        nextAtomEquivalent_features_pooled = torch_scatter.scatter_mean(nextAtomEquivalent_features, masked_next_atom_attachment_batch_index_reindexed[masked_multihot_next_attachments], dim = 0)
        graph_subgraph_focal_nextAtomFragment_focalAtom_nextAtom_features_concat_masked = torch.cat((graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat_masked, nextAtomEquivalent_features_pooled), dim = 1)
        bond_types_logits = self.MLP_bond_types(graph_subgraph_focal_nextAtomFragment_focalAtom_nextAtom_features_concat_masked)
        
        return stop_logits, next_atom_fragment_logits, fragment_attachment_scores_softmax, next_fragment_attachment_scores_softmax, bond_types_logits

    def decode_stop(self, graph_subgraph_focal_features_concat):
        stop_logits = self.MLP_stop_scores(graph_subgraph_focal_features_concat)
        return stop_logits

    def decode_next_atom_fragment(self, graph_subgraph_focal_focalAtom_features_concat_masked, fragment_library_features):
        graph_subgraph_focal_focalAtom_features_concat_masked_expand = graph_subgraph_focal_focalAtom_features_concat_masked.unsqueeze(0).expand((fragment_library_features.shape[0], -1, -1))
        fragment_library_features_expand = fragment_library_features.unsqueeze(0).expand((graph_subgraph_focal_focalAtom_features_concat_masked.shape[0], -1, -1)).permute(1,0,2)
        scoring_next_fragment_features = torch.cat((graph_subgraph_focal_focalAtom_features_concat_masked_expand, fragment_library_features_expand), dim = 2) 
        next_atom_fragment_logits = self.MLP_next_fragment_scores(scoring_next_fragment_features).squeeze(2).permute(1,0)
        return next_atom_fragment_logits


    def decode_focal_attachment_point(self, graph_subgraph_focal_features_concat, h_focal, focal_batch_index_reindexed):
        # ONLY AT INFERENCE
        graph_subgraph_focal_focalAttachments_features_concat = torch.cat((graph_subgraph_focal_features_concat[focal_batch_index_reindexed], h_focal), dim = 1)
        fragment_attachment_scores = self.MLP_fragment_attachment_scores(graph_subgraph_focal_focalAttachments_features_concat)
        fragment_attachment_scores_softmax = torch_scatter.composite.scatter_softmax(fragment_attachment_scores.squeeze(1), focal_batch_index_reindexed)
        return fragment_attachment_scores_softmax

    def decode_next_attachment_point(self, graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat, fragment_library_node_features, next_atom_attachment_indices, next_atom_attachment_batch_index_reindexed):
        # ONLY AT INFERENCE
        next_fragment_attachment_features = fragment_library_node_features[next_atom_attachment_indices]
        graph_subgraph_focal_nextAtomFragment_focalAtom_nextFragmentAttachments_features_concat = torch.cat((graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat[next_atom_attachment_batch_index_reindexed], next_fragment_attachment_features), dim = 1)
        next_fragment_attachment_scores = self.MLP_next_fragment_attachment_scores(graph_subgraph_focal_nextAtomFragment_focalAtom_nextFragmentAttachments_features_concat)
        next_fragment_attachment_scores_softmax = torch_scatter.composite.scatter_softmax(next_fragment_attachment_scores.squeeze(1), next_atom_attachment_batch_index_reindexed)
        return next_fragment_attachment_scores_softmax

    def decode_bond_type(self, graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat, multihot_next_attachments, next_atom_attachment_batch_index_reindexed, fragment_library_node_features, next_atom_attachment_indices):
        # ONLY AT INFERENCE
        nextAtomEquivalent_features = fragment_library_node_features[next_atom_attachment_indices[torch.tensor(multihot_next_attachments, dtype = torch.bool)]]
        nextAtomEquivalent_features_pooled = torch_scatter.scatter_mean(nextAtomEquivalent_features, next_atom_attachment_batch_index_reindexed[torch.tensor(multihot_next_attachments, dtype = torch.bool)], dim = 0)
        graph_subgraph_focal_nextAtomFragment_focalAtom_nextAtom_features_concat = torch.cat((graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat, nextAtomEquivalent_features_pooled), dim = 1)
        bond_types_logits = self.MLP_bond_types(graph_subgraph_focal_nextAtomFragment_focalAtom_nextAtom_features_concat)
        return bond_types_logits

        
