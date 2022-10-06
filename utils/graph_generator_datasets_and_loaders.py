import torch_geometric
import torch
import torch_scatter

import math
import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
import random
from tqdm import tqdm

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
from rdkit.Chem import rdMolTransforms
import rdkit.Chem.rdMolAlign
from rdkit.Geometry import Point3D
import rdkit.Chem.rdShapeHelpers
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolAlign


import collections
from collections.abc import Mapping, Sequence
from typing import List, Optional, Union
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData


from .general_utils import *

# This version (different from the one in general_utils.py) uses numpy arrays rather than pytorch tensors, to prevent memory leaks
def get_point_cloud_numpy(centers, N, per_node = True, var = 1./(12.*1.7)): # var = 1./(4.*1.7) |or| var = 1./(12.*1.7)
    if per_node:
        N_points_per_atom = N
    else:
        N_total = N
        N_points_per_atom = int(math.ceil(N_total / (centers.shape[0] - 1)))
                
    volume = []
    for center in centers:
        points = sample_atom_volume(center, N_points_per_atom, var = var)
        volume_points = points
        volume.append(points)
    
    cloud_batch_index_all =  np.concatenate([np.ones(N_points_per_atom, dtype = int) * i for i in range(centers.shape[0])], axis = 0)
    cloud_all = np.concatenate(volume)
    
    if per_node == False:
        subsamples = torch.LongTensor(np.sort(np.random.choice(np.arange(0, cloud_all.shape[0]), N_total, replace = False)))
        cloud = cloud_all[subsamples]
        cloud_batch_index = cloud_batch_index_all[subsamples]
    else:
        cloud = cloud_all
        cloud_batch_index = cloud_batch_index_all
    
    return cloud, cloud_batch_index

def sample_atom_volume(center, N, var = 1./(12.*1.7)):
    x = np.random.multivariate_normal(center, [[var, 0, 0], [0, var, 0], [0, 0, var]], size=N, check_valid='warn', tol=1e-8)
    return x


class VNNBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, db, max_batch_size, chunks = 1):
        self.db = db

        self.chunks = chunks
        self.max_batch_size = max_batch_size
        
        self.groups = self.db.groupby(['N_atoms', 'N_atoms_partial'])
        self.group_lengths = [len(g[1]) for g in self.groups]
        
    def __iter__(self): # returns list of lists of indices, with each inner list containing a batch of indices
        group_indices = [random.sample(list(g[1].index), k = len(g[1])) for g in self.groups]
        batch_indices = [[group[i:i+self.max_batch_size] for i in range(0, len(group), self.max_batch_size)] for group in group_indices]
        
        flattened_batches = [item for sublist in batch_indices for item in sublist]
        
        np.random.shuffle(flattened_batches)
        
        N = len(flattened_batches)
        flattened_batches = flattened_batches[0: int(N / self.chunks)]
        
        return iter(flattened_batches)

    def __len__(self): # number of batches
        return sum([math.ceil(n_g / self.max_batch_size) for n_g in self.group_lengths]) // self.chunks


class Collater:
    def __init__(self, follow_batch, exclude_keys, N_fragment_library_nodes, fragment_batch_batch):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.N_fragment_library_nodes = N_fragment_library_nodes
        self.fragment_batch_batch = fragment_batch_batch

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return_batch = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
            batch_dict = self.process_batch(return_batch)
            return return_batch, batch_dict

        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)

    def process_batch(self, batch):
        
        batch_size = batch.subgraph_size.shape[0]
    
        focal_index_rel_partial = batch.focal_index_subgraph
        focal_indices_batch = batch.x_subgraph_batch[batch.focal_index_subgraph]
        focal_attachment_index_rel_partial = batch.focal_attachment_index_subgraph
        
        stop_mask = batch.stop_mask
        stop_focal_mask = batch.stop_focal_mask
        focal_attachment_label_prob_masked = batch.focal_attachment_point_label_prob[batch.stop_focal_mask]
        all_stop = True not in stop_mask
            
        masked_focal_batch_index = batch.x_subgraph_batch[batch.focal_index_subgraph][batch.stop_focal_mask]
        focal_reindexing_map = {int(j):int(i) for i,j in zip(torch.arange(torch.unique(masked_focal_batch_index).shape[0]), torch.unique(masked_focal_batch_index))}
        masked_focal_batch_index_reindexed = torch.tensor([focal_reindexing_map[int(i)] for i in masked_focal_batch_index], dtype = torch.long)
        

        if not all_stop:
            next_atom_attachment_indices = torch.cat([torch.arange(self.N_fragment_library_nodes)[self.fragment_batch_batch == batch.next_atom_type_library_idx[i]] for i in range(batch.next_atom_type_library_idx.shape[0]) if stop_mask[i]])
            masked_next_atom_attachment_batch_index = torch.tensor(np.concatenate([[i]*len(batch.multi_hot_next_atom_fragment_attachment_points[i]) for i in range(len(batch.multi_hot_next_atom_fragment_attachment_points)) if stop_mask[i]]), dtype = torch.long)
            next_atom_attachment_reindexing_map = {int(j):int(i) for i,j in zip(torch.arange(torch.unique(masked_next_atom_attachment_batch_index).shape[0]), torch.unique(masked_next_atom_attachment_batch_index))}
            masked_next_atom_attachment_batch_index_reindexed = torch.tensor([next_atom_attachment_reindexing_map[int(i)] for i in masked_next_atom_attachment_batch_index], dtype = torch.long)
            
            masked_multihot_next_attachments_label_prob = [el for item in [j for i,j in enumerate(batch.next_atom_fragment_attachment_point_label_prob) if (stop_mask[i] == 1)] for el in item]
            masked_multihot_next_attachments = torch.tensor([el for item in [j for i,j in enumerate(batch.multi_hot_next_atom_fragment_attachment_points) if (stop_mask[i] == 1)] for el in item], dtype = torch.bool)
            
            next_atomFragment_attachment_loss_mask = torch.tensor([int(sum(masked_next_atom_attachment_batch_index_reindexed==i) > 1) for i in range(torch.unique(masked_next_atom_attachment_batch_index_reindexed).shape[0])], dtype = torch.bool)
            next_atomFragment_attachment_loss_mask_size = int(sum(next_atomFragment_attachment_loss_mask))
            
            if next_atomFragment_attachment_loss_mask_size > 0:
                mask_multi = torch.sum(torch.cat([(masked_next_atom_attachment_batch_index_reindexed == i).unsqueeze(0) for i in torch.unique(masked_next_atom_attachment_batch_index_reindexed)[next_atomFragment_attachment_loss_mask]], dim = 0), dim = 0) > 0
                select_multi_losses = torch.arange(0, masked_next_atom_attachment_batch_index_reindexed.shape[0])[mask_multi][torch.tensor(np.concatenate([i > 0. for i in batch.next_atom_fragment_attachment_point_label_prob if len(i) > 1 ])).reshape(-1)]
                temp_mask = torch.tensor(np.concatenate([(i > 0.)*n for n, i in enumerate(batch.next_atom_fragment_attachment_point_label_prob) if len(i) > 1 ]))
                temp_mask_bool = torch.BoolTensor(np.concatenate([i > 0. for i in batch.next_atom_fragment_attachment_point_label_prob if len(i) > 1 ]))
                reindexing_temp = {int(j):int(i) for i,j in zip(torch.arange(torch.unique(temp_mask[temp_mask_bool]).shape[0]), torch.unique(temp_mask[temp_mask_bool]))}
                mask_select_multi = torch.tensor([reindexing_temp[int(i)] for i in temp_mask[temp_mask_bool]], dtype = torch.long)
            else:
                select_multi_losses = None
                mask_select_multi = None

            # only consider the losses where there is more than one attachment point in the focal atom/fragment (e.g., only consider focal FRAGMENTS)
            focal_attachment_loss_mask = torch.tensor([int(sum(masked_focal_batch_index_reindexed==i) > 1) for i in range(torch.unique(masked_focal_batch_index_reindexed).shape[0])], dtype = torch.bool)
            focal_attachment_loss_mask_size = int(sum(focal_attachment_loss_mask))

            bond_type_mask = batch.bond_type_class_idx_label[stop_mask]
        
        else:
            next_atomFragment_attachment_loss_mask_size = 0
            focal_attachment_loss_mask = None
            focal_attachment_loss_mask_size = None
            select_multi_losses = None
            mask_select_multi = None
            focal_attachment_label_prob_masked = None
            masked_focal_batch_index_reindexed = None
            focal_attachment_index_rel_partial = None
            next_atom_attachment_indices = None
            masked_next_atom_attachment_batch_index_reindexed = None
            masked_multihot_next_attachments_label_prob = None
            masked_multihot_next_attachments = None
            bond_type_mask = None

        # not needed
        #stop_mask = torch.tensor(stop_mask, dtype = torch.bool)
        #stop_focal_mask = torch.tensor(stop_focal_mask, dtype = torch.bool)
        
        
        # computing tanimoto similarity between pairs of molecules in the batch
        fps_batch = batch.fingerprint.reshape(-1, 2048).numpy()
        indices_1 = np.random.choice(fps_batch.shape[0], 5000)
        indices_2 = np.random.choice(fps_batch.shape[0], 5000)
        A_and_B = np.sum(fps_batch[indices_1]&fps_batch[indices_2], axis = 1)
        A_plus_B = fps_batch[indices_1].sum(axis = 1) +fps_batch[indices_2].sum(axis = 1)
        sim = A_and_B / (A_plus_B - A_and_B)
        values = np.random.uniform(low=0.0, high=1.0, size=500) # approximate uniform sampling over tanimoto similarity values
        differences = (values.reshape(1,-1) - sim.reshape(-1,1))
        indices_ = np.abs(differences).argmin(axis=0)
        pairwise_indices_1_select = torch.from_numpy(indices_1[np.unique(indices_)]).type(torch.long)
        pairwise_indices_2_select = torch.from_numpy(indices_2[np.unique(indices_)]).type(torch.long)
        pairwise_targets = torch.from_numpy(sim[np.unique(indices_)]).float() # tanimoto similarity
        
        batch_dict = {}
        batch_dict['batch_size'] = batch_size

        batch_dict['focal_index_rel_partial'] = focal_index_rel_partial
        batch_dict['focal_indices_batch'] = focal_indices_batch
        
        batch_dict['stop_mask'] = stop_mask
        batch_dict['stop_focal_mask'] = stop_focal_mask
        batch_dict['all_stop'] = all_stop
        
        batch_dict['next_atomFragment_attachment_loss_mask_size'] = next_atomFragment_attachment_loss_mask_size
        batch_dict['focal_attachment_loss_mask'] = focal_attachment_loss_mask
        batch_dict['focal_attachment_loss_mask_size'] = focal_attachment_loss_mask_size
        batch_dict['select_multi_losses'] = select_multi_losses
        batch_dict['mask_select_multi'] = mask_select_multi
        batch_dict['focal_attachment_label_prob_masked'] = focal_attachment_label_prob_masked
        batch_dict['masked_focal_batch_index_reindexed'] = masked_focal_batch_index_reindexed
        batch_dict['focal_attachment_index_rel_partial'] = focal_attachment_index_rel_partial
        batch_dict['next_atom_attachment_indices'] = next_atom_attachment_indices
        batch_dict['masked_next_atom_attachment_batch_index_reindexed'] = masked_next_atom_attachment_batch_index_reindexed
        batch_dict['masked_multihot_next_attachments'] = masked_multihot_next_attachments
        batch_dict['masked_multihot_next_attachments_label_prob'] = masked_multihot_next_attachments_label_prob

        batch_dict['bond_type_mask'] = bond_type_mask
        
        batch_dict['pairwise_indices_1_select'] = pairwise_indices_1_select
        batch_dict['pairwise_indices_2_select'] = pairwise_indices_2_select
        batch_dict['pairwise_targets'] = pairwise_targets
        

        return batch_dict

class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,

        N_fragment_library_nodes = None, 
        fragment_batch_batch = None,

        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys, N_fragment_library_nodes, fragment_batch_batch),
            **kwargs,
        )
    

class PairData(torch_geometric.data.Data):
    def __init__(self, \
                 x=None, \
                 edge_index=None, \
                 edge_attr=None, \
                 pos=None, \
                 cloud=None, \
                 cloud_index=None, \
                 cloud_indices=None, \
                 atom_fragment_associations=None, \
                 subgraph_node_index=None, \
                 edge_index_subgraph=None, \
                 x_subgraph=None, \
                 edge_attr_subgraph=None, \
                 pos_subgraph=None, \
                 cloud_subgraph=None, \
                 cloud_index_subgraph=None, \
                 cloud_indices_subgraph=None, \
                 atom_fragment_associations_subgraph=None, \
                 focal_index=None, \
                 subgraph_size=None, \
                 focal_size=None, \
                 # should be same as focal_index_subgraph once indexed properly
                 focal_index_subgraph=None, \
                 focal_reindices = None, \
                 
                 stop_mask = None, \
                 stop_focal_mask = None, \
                 focal_attachment_index_subgraph = None, \
                 
                 next_atom_type_library_idx = None, \
                 focal_attachment_point_label_prob = None, \
                 next_atom_fragment_attachment_point_label_prob = None, \
                 multi_hot_next_atom_fragment_attachment_points = None, \
                 bond_type_class_idx_label = None, \
                 involves_fragment = None, \
                 
                 N_future_atoms = None, \
                 
                 fingerprint = None, \
                 mol_prop = None, \

                ):
        
        super().__init__()
        
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.cloud = cloud
        self.cloud_index = cloud_index
        self.cloud_indices = cloud_indices
        self.atom_fragment_associations = atom_fragment_associations
        
        self.subgraph_node_index = subgraph_node_index
        
        self.edge_index_subgraph = edge_index_subgraph
        self.x_subgraph = x_subgraph
        self.edge_attr_subgraph = edge_attr_subgraph
        self.pos_subgraph = pos_subgraph
        self.cloud_subgraph = cloud_subgraph
        self.cloud_index_subgraph = cloud_index_subgraph
        self.cloud_indices_subgraph = cloud_indices_subgraph
        self.atom_fragment_associations_subgraph = atom_fragment_associations_subgraph
        
        self.focal_index = focal_index
        self.subgraph_size = subgraph_size
        self.focal_size = focal_size
        self.focal_index_subgraph = focal_index_subgraph
        self.focal_reindices = focal_reindices
        
        self.stop_mask = stop_mask
        self.stop_focal_mask = stop_focal_mask
        self.focal_attachment_index_subgraph = focal_attachment_index_subgraph
        
        self.next_atom_type_library_idx = next_atom_type_library_idx
        self.focal_attachment_point_label_prob = focal_attachment_point_label_prob
        self.next_atom_fragment_attachment_point_label_prob = next_atom_fragment_attachment_point_label_prob 
        self.multi_hot_next_atom_fragment_attachment_points = multi_hot_next_atom_fragment_attachment_points
        self.bond_type_class_idx_label = bond_type_class_idx_label
        self.involves_fragment = involves_fragment
        
        self.N_future_atoms = N_future_atoms
        
        self.fingerprint = fingerprint
        self.mol_prop = mol_prop
        
    
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'cloud_index', 'subgraph_node_index', 'focal_index']:
            return self.x.size(0)
        if key in ['edge_index_subgraph', 'cloud_index_subgraph', 'focal_index_subgraph', 'focal_attachment_index_subgraph']:
            return self.x_subgraph.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)



class FragmentGraphDataset_point_cloud(torch_geometric.data.Dataset):
    def __init__(self, mols, original_index, edge_index_array, edge_features_array, node_features_array, xyz_array, atom_fragment_associations_array, atoms_pointer, bonds_pointer, focal_attachment_index, next_atom_index, partial_graph_indices_sorted, partial_graph_indices_sorted_pointer, focal_indices_sorted, focal_indices_sorted_pointer, next_atom_fragment_indices_sorted, next_atom_fragment_indices_sorted_pointer, focal_attachment_index_ref_partial_array, focal_attachment_point_label_prob_array, focal_attachment_point_label_prob_pointer, multi_hot_next_atom_fragment_attachment_points_array, multi_hot_next_atom_fragment_attachment_points_pointer, bond_type_class_index_label_array, N_points = 10, dihedral_var = 0.0, xyz_var = 0.0, randomize_focal_dihedral = False):
        super(FragmentGraphDataset_point_cloud, self).__init__()

        self.mols = mols # None or list of mols accessible by original_index (NOT key)

        self.N_points = N_points

        self.original_index = original_index 

        self.edge_index_array = edge_index_array
        self.edge_features_array = edge_features_array
        self.node_features_array = node_features_array
        self.xyz_array = xyz_array
        self.atom_fragment_associations_array = atom_fragment_associations_array
        self.atoms_pointer = atoms_pointer
        self.bonds_pointer = bonds_pointer

        self.focal_attachment_index = focal_attachment_index 
        self.next_atom_index = next_atom_index 
        
        self.partial_graph_indices_sorted = partial_graph_indices_sorted 
        self.partial_graph_indices_sorted_pointer = partial_graph_indices_sorted_pointer

        self.focal_indices_sorted = focal_indices_sorted 
        self.focal_indices_sorted_pointer = focal_indices_sorted_pointer

        self.next_atom_fragment_indices_sorted = next_atom_fragment_indices_sorted 
        self.next_atom_fragment_indices_sorted_pointer = next_atom_fragment_indices_sorted_pointer

        self.focal_attachment_index_ref_partial_array = focal_attachment_index_ref_partial_array
        self.focal_attachment_point_label_prob_array = focal_attachment_point_label_prob_array
        self.focal_attachment_point_label_prob_pointer = focal_attachment_point_label_prob_pointer
        self.multi_hot_next_atom_fragment_attachment_points_array = multi_hot_next_atom_fragment_attachment_points_array
        self.multi_hot_next_atom_fragment_attachment_points_pointer = multi_hot_next_atom_fragment_attachment_points_pointer
        self.bond_type_class_index_label_array = bond_type_class_index_label_array

        self.dihedral_var = dihedral_var
        self.xyz_var = xyz_var
        self.randomize_focal_dihedral = randomize_focal_dihedral
        
    def __len__(self):
        #return len(self.db)
        return len(self.original_index)
    
    def get_number_future_atoms(self, mol, next_atom_index, focal_indices):

        next_atom_neighbors = tuple([a.GetIdx() for a in mol.GetAtomWithIdx(int(next_atom_index)).GetNeighbors()])
        focal_attachment = int([f for f in focal_indices if (f in next_atom_neighbors)][0])
        bonds = [mol.GetBondBetweenAtoms(focal_attachment, int(next_atom_index)).GetIdx()]
        
        pieces = rdkit.Chem.FragmentOnSomeBonds(mol, bonds, numToBreak=len(bonds), addDummies=False) 
        disjoint_graphs = rdkit.Chem.GetMolFrags(pieces[0])
        
        future_atoms = disjoint_graphs[0] if (next_atom_index in disjoint_graphs[0]) else disjoint_graphs[1]
        
        return float(len(future_atoms))

    def augment_perturb_structure(self, mol, partial_indices, focal_indices, dihedral_var = 15, xyz_var = 0.1, randomize_focal_dihedral = True):
        all_rot_bonds = list(get_acyclic_single_bonds(mol))
        rot_bonds_partial = [r for r in all_rot_bonds if ((r[0] in partial_indices) & (r[1] in partial_indices))]
        dihedrals = []
        for bond_ in rot_bonds_partial:
            if int(bond_[0]) == int(focal_indices[0]):
                p1 = ()
            else:
                p1 = rdkit.Chem.rdmolops.GetShortestPath(mol, int(bond_[0]), int(focal_indices[0]))
            if int(bond_[1]) == int(focal_indices[0]):
                p2 = ()
            else:
                p2 = rdkit.Chem.rdmolops.GetShortestPath(mol, int(bond_[1]), int(focal_indices[0]))
            
            bond = tuple(reversed(bond_)) if len(p1) > len(p2) else bond_ # if bond[0] is farther away, then it should be moved
            
            first_neighbors = tuple(set([a.GetIdx() for a in mol.GetAtomWithIdx(bond[0]).GetNeighbors()]) - set([bond[1]]))
            first_neighbors = [f for f in first_neighbors if f in partial_indices]
            second_neighbors = tuple(set([a.GetIdx() for a in mol.GetAtomWithIdx(bond[1]).GetNeighbors()]) - set([bond[0]]))
            second_neighbors = [f for f in second_neighbors if f in partial_indices]
            if (len(first_neighbors) > 0) & (len(second_neighbors) > 0):
                d = (first_neighbors[0], *bond, second_neighbors[0])
                dihedrals.append(d)
        
        mol_augmented = deepcopy(mol)


        if dihedral_var > 0.0:
            angles_1 = np.random.normal(0, dihedral_var, len(dihedrals))
            
            for d_idx, d in enumerate(dihedrals):
                rdkit.Chem.rdMolTransforms.SetDihedralDeg(mol_augmented.GetConformer(), *d, float(angles_1[d_idx] + rdkit.Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *d)))

        if xyz_var > 0.0:
            xyz_augmented = mol_augmented.GetConformer().GetPositions()
            xyz_augmented = xyz_augmented + np.random.normal(0, xyz_var, (xyz_augmented.shape[0],3))
            for k in range(mol_augmented.GetNumAtoms()):
                x,y,z = np.array(xyz_augmented[k])
                mol_augmented.GetConformer().SetAtomPosition(k, Point3D(x,y,z))

        if randomize_focal_dihedral == True:
            focal_dihedral = [d for d in dihedrals if ((d[1] in partial_indices) & (d[2] in partial_indices) & ((d[1] in focal_indices) | (d[2] in focal_indices)))]
            if len(focal_dihedral) == 1:

                focal_dihedral_ = focal_dihedral[0]
                if focal_dihedral_[2] in focal_indices:
                    pass
                elif focal_dihedral_[1] in focal_indices:
                    focal_dihedral_ = tuple(reversed(focal_dihedral_))

                rdkit.Chem.rdMolTransforms.SetDihedralDeg(mol_augmented.GetConformer(), *focal_dihedral_, float(np.random.uniform(0, 360, 1)))

        
        return mol_augmented

    
    def __getitem__(self, key): 
        
        original_index = deepcopy(self.original_index[key])
        focal_attachment_index = deepcopy(self.focal_attachment_index[key])
        next_atom_index = deepcopy(self.next_atom_index[key])
        partial_graph_indices = deepcopy(self.partial_graph_indices_sorted[self.partial_graph_indices_sorted_pointer[key]:self.partial_graph_indices_sorted_pointer[key + 1]])
        focal_indices = deepcopy(self.focal_indices_sorted[self.focal_indices_sorted_pointer[key]:self.focal_indices_sorted_pointer[key + 1]]) 
        next_atom_fragment_indices = deepcopy(self.next_atom_fragment_indices_sorted[self.next_atom_fragment_indices_sorted_pointer[key]:self.next_atom_fragment_indices_sorted_pointer[key + 1]])

        # 0 / False indicates a stop token
        stop_mask = np.array([int(next_atom_index != -1)], dtype = bool) 
        
        involves_fragment = np.array([0], dtype = bool) 
        if (len(focal_indices) > 1):
            involves_fragment = np.array([1], dtype = bool) 
        elif next_atom_index != -1:
            if len(next_atom_fragment_indices) > 1:
                involves_fragment = np.array([1], dtype = bool)  

        # indices of focal atoms reference w.r.t. the partial subgraph
        focal_indices_ref_partial = [np.where(partial_graph_indices == f)[0][0] for f in focal_indices]

        focal_attachment_index_ref_partial = deepcopy(self.focal_attachment_index_ref_partial_array[key: key+1])
        focal_attachment_point_label_prob = deepcopy(self.focal_attachment_point_label_prob_array[self.focal_attachment_point_label_prob_pointer[key] : self.focal_attachment_point_label_prob_pointer[key+1]])

        # FULL GRAPH DATA
        edge_index = deepcopy(self.edge_index_array[:, self.bonds_pointer[original_index] : self.bonds_pointer[original_index+1]])
        edge_features = deepcopy(self.edge_features_array[self.bonds_pointer[original_index] : self.bonds_pointer[original_index+1]])
        node_features = deepcopy(self.node_features_array[self.atoms_pointer[original_index] : self.atoms_pointer[original_index+1]])
        atom_fragment_associations = deepcopy(self.atom_fragment_associations_array[self.atoms_pointer[original_index] : self.atoms_pointer[original_index+1]])
        
        if self.mols != None:
            mol = deepcopy(self.mols[original_index])
            
            fingerprint = _calc_fp_rdkit(mol)
            mol_prop = _calc_mol_property(mol, prop_name = 'QED')
            
            if (self.dihedral_var > 0.0) | (self.xyz_var > 0.0) | (self.randomize_focal_dihedral == True):
                mol_augmented = self.augment_perturb_structure(mol, partial_graph_indices, focal_indices, dihedral_var = self.dihedral_var, xyz_var = self.xyz_var, randomize_focal_dihedral = self.randomize_focal_dihedral)
            else:
                mol_augmented = mol
            
            positions_augmented = mol_augmented.GetConformer().GetPositions()


        xyz = deepcopy(self.xyz_array[self.atoms_pointer[original_index] : self.atoms_pointer[original_index+1]]) 
        center_of_mass = np.sum(xyz, axis = 0) / xyz.shape[0] 
        positions = xyz - center_of_mass

        if self.mols != None:
            positions_augmented = positions_augmented - center_of_mass # need to subtract the center of mass of the encoded target molecule.


        if next_atom_index != -1:
            next_atom_fragment_ID_index = np.array([atom_fragment_associations[next_atom_index]])
            multi_hot_next_atom_fragment_attachment_points = deepcopy(self.multi_hot_next_atom_fragment_attachment_points_array[self.multi_hot_next_atom_fragment_attachment_points_pointer[key] : self.multi_hot_next_atom_fragment_attachment_points_pointer[key + 1]])
            next_atom_fragment_attachment_point_label_prob = multi_hot_next_atom_fragment_attachment_points / sum(multi_hot_next_atom_fragment_attachment_points)
            bond_type_class_index_label = deepcopy(self.bond_type_class_index_label_array[key:key+1])
        else:
            next_atom_fragment_ID_index = np.array([-1]) # np.array([0])
            multi_hot_next_atom_fragment_attachment_points = [None]
            next_atom_fragment_attachment_point_label_prob = [None] 
            bond_type_class_index_label = np.array([-1])
            
        # stop shape penalty
        if self.mols != None:
            if next_atom_index != -1:
                num_future_atoms = self.get_number_future_atoms(mol, next_atom_index, focal_indices)
                N_future_atoms = np.array([num_future_atoms])
            else:
                N_future_atoms = np.array([0.0])
        else:
            N_future_atoms = np.array([0.0])
        

        # DATA COMMON TO ALL 2D PREDICTION TASKS
        subgraph_node_index = partial_graph_indices 
        
        n_idx =  np.zeros(positions.shape[0], dtype=int) 
        n_idx[subgraph_node_index] = np.arange(len(subgraph_node_index)) 
        
        edge_index_subgraph, edge_attr_subgraph = subgraph(
            subgraph_node_index, 
            edge_index, 
            edge_features,
            num_nodes = positions.shape[0],
            relabel_nodes = True,
        )
        
        if self.mols != None:
            positions_subgraph = positions_augmented[subgraph_node_index]
        else:
            positions_subgraph = positions[subgraph_node_index]
        
        atom_fragment_associations_subgraph = atom_fragment_associations[subgraph_node_index]
        
        subgraph_size = np.array([len(partial_graph_indices)])
        focal_size = np.array([len(focal_indices)])
        focal_index = focal_indices # this is indexed w.r.t full graph!
        
        cloud, cloud_batch_indices = get_point_cloud_numpy(positions, self.N_points, per_node = True)
        cloud_subgraph, cloud_batch_indices_subgraph = get_point_cloud_numpy(positions_subgraph, self.N_points, per_node = True)
        
        

        data = PairData(
            x = torch.from_numpy(node_features), # tensor
            edge_index = torch.from_numpy(edge_index).type(torch.long), # tensor
            edge_attr = torch.from_numpy(edge_features), # tensor
            pos = torch.from_numpy(positions), # tensor
            cloud = torch.from_numpy(cloud), # tensor
            cloud_index = torch.from_numpy(cloud_batch_indices).type(torch.long),
            cloud_indices = torch.from_numpy(cloud_batch_indices).type(torch.long),
            atom_fragment_associations = torch.from_numpy(atom_fragment_associations).type(torch.long), 
            
            subgraph_node_index = torch.from_numpy(subgraph_node_index).type(torch.long),
            
            edge_index_subgraph = torch.from_numpy(edge_index_subgraph).type(torch.long),
            x_subgraph = torch.from_numpy(node_features[subgraph_node_index]),
            edge_attr_subgraph = torch.from_numpy(edge_attr_subgraph),
            pos_subgraph = torch.from_numpy(positions_subgraph),
            cloud_subgraph = torch.from_numpy(cloud_subgraph),
            cloud_index_subgraph = torch.from_numpy(cloud_batch_indices_subgraph).type(torch.long),
            cloud_indices_subgraph = torch.from_numpy(cloud_batch_indices_subgraph).type(torch.long),
            atom_fragment_associations_subgraph = torch.from_numpy(atom_fragment_associations_subgraph).type(torch.long), 
            
            focal_index = torch.from_numpy(focal_index).type(torch.long),
            subgraph_size = torch.from_numpy(subgraph_size),
            focal_size = torch.from_numpy(focal_size),
            focal_index_subgraph = torch.from_numpy(n_idx[focal_index]).type(torch.long),
            focal_reindices = torch.from_numpy(np.array(focal_indices_ref_partial)).type(torch.long),
            
            stop_mask = torch.from_numpy(stop_mask).type(torch.bool),
            stop_focal_mask = torch.from_numpy(stop_mask.repeat(len(focal_indices_ref_partial))).type(torch.bool), 
            focal_attachment_index_subgraph = torch.from_numpy(focal_attachment_index_ref_partial).type(torch.long),
            
            next_atom_type_library_idx = torch.from_numpy(next_atom_fragment_ID_index).type(torch.long),
            focal_attachment_point_label_prob = torch.from_numpy(focal_attachment_point_label_prob),
            next_atom_fragment_attachment_point_label_prob = next_atom_fragment_attachment_point_label_prob,
            multi_hot_next_atom_fragment_attachment_points = multi_hot_next_atom_fragment_attachment_points,
            bond_type_class_idx_label = torch.from_numpy(bond_type_class_index_label),
            involves_fragment = torch.from_numpy(involves_fragment).type(torch.bool),
            
            N_future_atoms = torch.from_numpy(N_future_atoms),
            fingerprint = torch.from_numpy(fingerprint).type(torch.uint8),
            mol_prop = torch.from_numpy(mol_prop),

        )

        return data
    

def subgraph(subset, edge_index, edge_attr, num_nodes, relabel_nodes=False):
    # numpy implementation of torch_geometric version
    n_mask = np.zeros(num_nodes, dtype=int)
    n_mask[subset] = 1

    if relabel_nodes:
        n_idx = np.zeros(num_nodes, dtype=int)
        n_idx[subset] = np.arange(len(subset))

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr

def _calc_fp_rdkit(mol):
    fp_fn = lambda m: rdkit.Chem.RDKFingerprint(m)
    fingerprint = fp_fn(mol)
    array = np.zeros((0,), dtype=np.int8)
    rdkit.Chem.DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array

import rdkit.Chem.QED
def _calc_mol_property(mol, prop_name = None):
    if prop_name == 'QED':
        prop = rdkit.Chem.QED.default(mol)
    else:
        prop = 0.
    return np.array([prop])


