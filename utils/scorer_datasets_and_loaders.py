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


class ROCS_PairData(torch_geometric.data.Data):
    def __init__(self, \
            x = None, \
            edge_index = None, \
            edge_attr = None, \
            pos = None, \
            cloud = None, \
            cloud_indices = None, \
            atom_fragment_associations = None, \
            
            x_subgraph = None, \
            edge_index_subgraph = None, \
            subgraph_node_index = None, \
            edge_attr_subgraph = None, \
            pos_subgraph = None, \
            cloud_subgraph = None, \
            cloud_indices_subgraph = None, \
            atom_fragment_associations_subgraph = None, \

            query_index_subgraph = None, \
            subgraph_size = None, \
            query_size = None, \
                 
            new_batch = None, \
            new_batch_subgraph = None, \
        ):
        
        super().__init__()
        
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.cloud = cloud
        self.cloud_indices = cloud_indices
        self.atom_fragment_associations = atom_fragment_associations

        self.x_subgraph = x_subgraph
        self.edge_index_subgraph = edge_index_subgraph
        self.subgraph_node_index = subgraph_node_index
        self.edge_attr_subgraph = edge_attr_subgraph
        self.pos_subgraph = pos_subgraph
        self.cloud_subgraph = cloud_subgraph
        self.cloud_indices_subgraph = cloud_indices_subgraph
        self.atom_fragment_associations_subgraph = atom_fragment_associations_subgraph

        self.query_index_subgraph = query_index_subgraph
        self.subgraph_size = subgraph_size
        self.query_size = query_size
        self.new_batch = new_batch
        self.new_batch_subgraph = new_batch_subgraph
    
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'subgraph_node_index']:
            return self.x.size(0)
        if key in ['edge_index_subgraph', 'query_index_subgraph']:
            return self.x_subgraph.size(0)
        if key in ['new_batch']:
            return int(max(self.new_batch) + 1)
        if key in ['new_batch_subgraph']:
            return int(max(self.new_batch_subgraph) + 1)
        else:
            return super().__inc__(key, value, *args, **kwargs)


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


class ROCSDataset_point_cloud(torch_geometric.data.Dataset):
    def __init__(self, mols, max_future_rocs, max_future_rocs_evaluated_dihedrals, max_future_rocs_index, original_index, edge_index_array, edge_features_array, node_features_array, xyz_array, atom_fragment_associations_array, atoms_pointer, bonds_pointer, dihedral_indices_array, dihedral_indices_pointer, indices_partial_before_array, indices_partial_before_pointer, indices_partial_after_array, indices_partial_after_pointer, query_indices_array, query_indices_pointer, N_points, N_rot, dihedral_var = 0.0):
        super(ROCSDataset_point_cloud, self).__init__()
        self.mols = mols # specific to training/validatoin splits

        self.max_future_rocs = max_future_rocs # for ALL data
        self.max_future_rocs_evaluated_dihedrals = max_future_rocs_evaluated_dihedrals # for ALL data
        self.max_future_rocs_index = max_future_rocs_index # specific to training/validation splits


        self.N_rot = N_rot
        self.N_points = N_points

        self.original_index = original_index

        self.edge_index_array = edge_index_array
        self.edge_features_array = edge_features_array
        self.node_features_array = node_features_array
        self.xyz_array = xyz_array
        self.atom_fragment_associations_array = atom_fragment_associations_array
        self.atoms_pointer = atoms_pointer
        self.bonds_pointer = bonds_pointer
        
        self.dihedral_indices_array = dihedral_indices_array
        self.dihedral_indices_pointer = dihedral_indices_pointer
        self.indices_partial_before_array = indices_partial_before_array
        self.indices_partial_before_pointer = indices_partial_before_pointer
        self.indices_partial_after_array = indices_partial_after_array
        self.indices_partial_after_pointer  = indices_partial_after_pointer
        self.query_indices_array = query_indices_array
        self.query_indices_pointer = query_indices_pointer

        self.dihedral_var = dihedral_var

        
    def __len__(self):
        return len(self.mols)

    def augment_perturb_structure(self, mol, partial_indices, dihedral_indices, dihedral_var = 15.0):
        all_rot_bonds = list(get_acyclic_single_bonds(mol))
        rot_bonds_partial = [r for r in all_rot_bonds if ((r[0] in partial_indices) & (r[1] in partial_indices))]
        dihedrals = []
        for bond_ in rot_bonds_partial:
            if int(bond_[0]) == int(dihedral_indices[3]):
                p1 = ()
            else:
                p1 = rdkit.Chem.rdmolops.GetShortestPath(mol, int(bond_[0]), int(dihedral_indices[3]))
            if int(bond_[1]) == int(dihedral_indices[3]):
                p2 = ()
            else:
                p2 = rdkit.Chem.rdmolops.GetShortestPath(mol, int(bond_[1]), int(dihedral_indices[3]))
            
            bond = tuple(reversed(bond_)) if len(p1) > len(p2) else bond_ # if bond[0] is farther away, then it should be moved
            
            first_neighbors = tuple(set([a.GetIdx() for a in mol.GetAtomWithIdx(bond[0]).GetNeighbors()]) - set([bond[1]]))
            first_neighbors = [f for f in first_neighbors if f in partial_indices]
            second_neighbors = tuple(set([a.GetIdx() for a in mol.GetAtomWithIdx(bond[1]).GetNeighbors()]) - set([bond[0]]))
            second_neighbors = [f for f in second_neighbors if f in partial_indices]
            if (len(first_neighbors) > 0) & (len(second_neighbors) > 0) & (sorted(dihedral_indices[1:3]) != sorted(bond)):
                d = (first_neighbors[0], *bond, second_neighbors[0])
                dihedrals.append(d)
        #also include the query dihedral, but in its reversed ordering...
        dihedrals.append(tuple(reversed(dihedral_indices)))

        mol_augmented = deepcopy(mol)

        if dihedral_var > 0.0:
            angles_1 = np.random.normal(0, dihedral_var, len(dihedrals))
            for d_idx, d in enumerate(dihedrals):
                rdkit.Chem.rdMolTransforms.SetDihedralDeg(mol_augmented.GetConformer(), *d, float(angles_1[d_idx] + rdkit.Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *d)))

        return mol_augmented

    
    def __getitem__(self, key):

        mol = deepcopy(self.mols[key])
        
        original_index = deepcopy(self.original_index[key])

        max_future_rocs_index = deepcopy(self.max_future_rocs_index[key])
        max_future_rocs_array = deepcopy(self.max_future_rocs[max_future_rocs_index])
        max_future_rocs_evaluated_dihedrals_array = deepcopy(self.max_future_rocs_evaluated_dihedrals[max_future_rocs_index])
        
        dihedral_indices = deepcopy(self.dihedral_indices_array[self.dihedral_indices_pointer[key]:self.dihedral_indices_pointer[key + 1]])
        dihedral_indices = tuple([int(d) for d in dihedral_indices])

        indices_partial_before = deepcopy(self.indices_partial_before_array[self.indices_partial_before_pointer[key]:self.indices_partial_before_pointer[key + 1]])
        indices_partial_after = deepcopy(self.indices_partial_after_array[self.indices_partial_after_pointer[key]:self.indices_partial_after_pointer[key + 1]])
        query_indices = deepcopy(self.query_indices_array[self.query_indices_pointer[key]:self.query_indices_pointer[key + 1]])

        if dihedral_indices[3] in indices_partial_before:
            dihedral_indices = tuple(reversed(dihedral_indices))
        assert dihedral_indices[3] not in indices_partial_before 

        # FULL GRAPH DATA
        edge_index = deepcopy(self.edge_index_array[:, self.bonds_pointer[original_index] : self.bonds_pointer[original_index+1]])
        edge_features = deepcopy(self.edge_features_array[self.bonds_pointer[original_index] : self.bonds_pointer[original_index+1]])
        node_features = deepcopy(self.node_features_array[self.atoms_pointer[original_index] : self.atoms_pointer[original_index+1]])
        atom_fragment_associations = deepcopy(self.atom_fragment_associations_array[self.atoms_pointer[original_index] : self.atoms_pointer[original_index+1]])
        
        xyz = deepcopy(self.xyz_array[self.atoms_pointer[original_index] : self.atoms_pointer[original_index+1]]) 
        center_of_mass = np.sum(xyz, axis = 0) / xyz.shape[0] 
        positions = xyz - center_of_mass

        subgraph_node_index = indices_partial_after
        
        n_idx = np.zeros(positions.shape[0], dtype=int)
        n_idx[subgraph_node_index] = np.arange(len(subgraph_node_index))
        
        edge_index_subgraph, edge_attr_subgraph = subgraph(
            subgraph_node_index, 
            edge_index, 
            edge_features,
            num_nodes = positions.shape[0],
            relabel_nodes = True,
        )

        G = deepcopy(get_substructure_graph(mol, list(range(0, mol.GetNumAtoms()))))
        G.remove_edge(dihedral_indices[1], dihedral_indices[2])
        disjoint_graphs = [list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)]
        assert len(disjoint_graphs) == 2
        unknown_positions = sorted(disjoint_graphs[1]) if len(set(indices_partial_before).intersection(set(disjoint_graphs[0]))) > len(set(indices_partial_before).intersection(set(disjoint_graphs[1]))) else sorted(disjoint_graphs[0]) # this should ONLY contain the indices of nodes whose positions are influenced by the dihedral (as well as the focal root node)

        N = deepcopy(self.N_rot)
        rotated_partial_positions_list = []
        
        selected_dihedrals = np.random.choice(range(1, max_future_rocs_evaluated_dihedrals_array.shape[0]), size = N-1, replace = False)
        selected_dihedrals = np.concatenate((np.array([0]), selected_dihedrals))

        rots = max_future_rocs_evaluated_dihedrals_array[selected_dihedrals]
        future_ROCS = max_future_rocs_array[selected_dihedrals]

        # DATA AUGMENTATION OF PARTIAL GRAPH
        mol_augmented = self.augment_perturb_structure(mol, indices_partial_before, dihedral_indices, dihedral_var = self.dihedral_var)

        original_mol = deepcopy(mol_augmented) 
        for r, rot in enumerate(rots):
            mol_rotated = deepcopy(original_mol) 
            rdkit.Chem.rdMolTransforms.SetDihedralDeg(mol_rotated.GetConformer(), *dihedral_indices, rot)

            rotated_pos = np.array(mol_rotated.GetConformer().GetPositions())
            rotated_partial_positions_list.append(rotated_pos[subgraph_node_index] - center_of_mass)

        rotated_partial_positions = np.concatenate(rotated_partial_positions_list, axis = 0)        

        x = node_features
        edge_attr = edge_features
        pos = positions
        x_subgraph = x[subgraph_node_index]
        
        atom_fragment_associations_subgraph = atom_fragment_associations[subgraph_node_index]
        query_index_subgraph = n_idx[query_indices] 
        
        subgraph_size = np.array([len(indices_partial_after)])
        query_size = np.array([len(query_indices)])
        

        add_to_edge_index = np.concatenate([np.ones(edge_index.shape, dtype = int)*i for i in range(N)], axis = 1) * x.shape[0]
        add_to_subgraph_node_index = np.concatenate([np.ones(subgraph_node_index.shape, dtype = int)*i for i in range(N)], axis = 0) * x.shape[0]
        atom_fragment_associations_repeat = np.tile(atom_fragment_associations, N)
        x_repeat = np.tile(x, (N,1))
        edge_index_repeat = np.tile(edge_index, (1,N)) + add_to_edge_index
        edge_attr_repeat = np.tile(edge_attr, (N,1))
        pos_repeat = np.tile(pos, (N,1))
        
        cloud_temp_list = [get_point_cloud_numpy(pos, self.N_points, per_node = True) for i in range(N)]
        cloud_list, cloud_batch_indices_list = [t[0] for t in cloud_temp_list], [t[1] for t in cloud_temp_list]
        cloud = np.concatenate(cloud_list, axis = 0)
        cloud_batch_indices = np.concatenate(cloud_batch_indices_list, axis = 0)
        
        add_to_edge_index_subgraph = np.concatenate([np.ones(edge_index_subgraph.shape, dtype = int)*i for i in range(N)], axis = 1) * x_subgraph.shape[0]
        add_to_query_index_subgraph = np.concatenate([np.ones(query_index_subgraph.shape, dtype = int)*i for i in range(N)], axis = 0) * x_subgraph.shape[0]
        
        pos_subgraph_repeat = rotated_partial_positions
        x_subgraph_repeat = np.tile(x_subgraph, (N,1))
        edge_index_subgraph_repeat = np.tile(edge_index_subgraph, (1,N)) + add_to_edge_index_subgraph
        subgraph_node_index_repeat = np.tile(subgraph_node_index, N) + add_to_subgraph_node_index 
        edge_attr_subgraph_repeat = np.tile(edge_attr_subgraph, (N,1))
        query_index_subgraph_repeat = np.tile(query_index_subgraph, N) + add_to_query_index_subgraph 
        atom_fragment_associations_subgraph_repeat = np.tile(atom_fragment_associations_subgraph, N)
        
        subgraph_size_repeat = np.tile(subgraph_size, N)
        query_size_repeat = np.tile(query_size, N)

        cloud_subgraph_temp_list = [get_point_cloud_numpy(p, self.N_points, per_node = True) for p in rotated_partial_positions_list]
        cloud_subgraph_list, cloud_batch_indices_subgraph_list = [t[0] for t in cloud_subgraph_temp_list], [t[1] for t in cloud_subgraph_temp_list]
        cloud_subgraph = np.concatenate(cloud_subgraph_list, axis = 0)
        cloud_batch_indices_subgraph = np.concatenate(cloud_batch_indices_subgraph_list, axis = 0)
        

        data = ROCS_PairData(
            x = torch.from_numpy(x_repeat), 
            edge_index = torch.from_numpy(edge_index_repeat).type(torch.long), 
            edge_attr = torch.from_numpy(edge_attr_repeat),
            pos = torch.from_numpy(pos_repeat),
            cloud = torch.from_numpy(cloud),
            cloud_indices = torch.from_numpy(cloud_batch_indices).type(torch.long),
            atom_fragment_associations = torch.from_numpy(atom_fragment_associations_repeat).type(torch.long),
            
            x_subgraph = torch.from_numpy(x_subgraph_repeat),
            edge_index_subgraph = torch.from_numpy(edge_index_subgraph_repeat).type(torch.long),
            subgraph_node_index = torch.from_numpy(subgraph_node_index_repeat).type(torch.long),
            edge_attr_subgraph = torch.from_numpy(edge_attr_subgraph_repeat),
            pos_subgraph = torch.from_numpy(pos_subgraph_repeat),
            cloud_subgraph = torch.from_numpy(cloud_subgraph),
            cloud_indices_subgraph = torch.from_numpy(cloud_batch_indices_subgraph).type(torch.long),
            atom_fragment_associations_subgraph = torch.from_numpy(atom_fragment_associations_subgraph_repeat).type(torch.long),

            query_index_subgraph = torch.from_numpy(query_index_subgraph_repeat).type(torch.long),
            subgraph_size = torch.from_numpy(subgraph_size_repeat),
            query_size = torch.from_numpy(query_size_repeat),
            
            new_batch = torch.from_numpy(np.concatenate([np.ones(x.shape[0], dtype = int)*i for i in range(N)], axis = 0)).type(torch.long),
            new_batch_subgraph = torch.from_numpy(np.concatenate([np.ones(x_subgraph.shape[0], dtype = int)*i for i in range(N)], axis = 0)).type(torch.long),
        )
        
        return data, torch.from_numpy(future_ROCS)

    