import torch

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import random
import networkx as nx
import os

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

def getNodeFeaturesForGraphMatching(list_rdkit_atoms):
    atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

    def one_hot_embedding(value, options):
        embedding = [0]*(len(options) + 1)
        index = options.index(value) if value in options else -1
        embedding[index] = 1
        return embedding
    
    F_v = (len(atomTypes)+1)
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes) # atom symbol, dim=12 + 1 
        
        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)


def get_substructure_graph_for_matching(mol, atom_indices, node_features = None):
    G = nx.Graph()
    bonds = list(mol.GetBonds())
    bond_indices = [sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]) for b in bonds]
    
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        if node_features is None:
            atom_features = getNodeFeaturesForGraphMatching([atom])[0]
        else:
            atom_features = node_features[atom_idx]
        G.add_node(atom_idx, atom_features = atom_features)
        
    for i in atom_indices:
        for j in atom_indices:
            if sorted([i,j]) in bond_indices:
                G.add_edge(i, j, bond_type=mol.GetBondBetweenAtoms(int(i), int(j)).GetBondTypeAsDouble())
    return G

def get_reindexing_map_for_matching(mol, fragment_indices, partial_mol):
    G1 = get_substructure_graph_for_matching(mol, fragment_indices)
    G2 = get_substructure_graph_for_matching(partial_mol, list(range(0, partial_mol.GetNumAtoms())))
    
    nm = nx.algorithms.isomorphism.generic_node_match(['atom_features'], [None], [np.allclose])
    em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
    
    # getting map from old indices to new indices
    GM = nx.algorithms.isomorphism.GraphMatcher(G1,
                                                G2, 
                                                node_match = nm,
                                                edge_match = em)
    assert GM.is_isomorphic() # THIS NEEDS TO BE CALLED FOR GM.mapping to be initiated
    idx_map = GM.mapping
    
    return idx_map

def set_mol_positions(mol, positions):
    mol_ = deepcopy(mol)
    for k in range(mol.GetNumAtoms()):
        x,y,z = np.array(positions[k])
        mol_.GetConformer().SetAtomPosition(k, Point3D(x,y,z))
    return mol_


def VAB_2nd_order_batched(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    R2 = ((torch.cdist(centers_1, centers_2)**2.0).T).permute(2,0,1)    
    prefactor1_prod_prefactor2 = (prefactors_1.unsqueeze(1) * prefactors_2.unsqueeze(2))
    alpha1_prod_alpha2 = (alphas_1.unsqueeze(1) * alphas_2.unsqueeze(2))
    alpha1_sum_alpha2 = (alphas_1.unsqueeze(1) + alphas_2.unsqueeze(2))    
    VAB_2nd_order = torch.sum(torch.sum(np.pi**(1.5) * prefactor1_prod_prefactor2 * torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2) / (alpha1_sum_alpha2**(1.5)), dim = 2), dim = 1)
    return VAB_2nd_order

def shape_tanimoto_batched(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    VAA = VAB_2nd_order_batched(centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1)
    VBB = VAB_2nd_order_batched(centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2)
    VAB = VAB_2nd_order_batched(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return VAB / (VAA + VBB - VAB)

def get_acyclic_single_bonds(mol):
    AcyclicBonds = rdkit.Chem.MolFromSmarts('[*]!@[*]')
    SingleBonds = rdkit.Chem.MolFromSmarts('[*]-[*]')
    acyclicBonds = mol.GetSubstructMatches(AcyclicBonds)
    singleBonds = mol.GetSubstructMatches(SingleBonds)
    
    acyclicBonds = [tuple(sorted(b)) for b in acyclicBonds]
    singleBonds = [tuple(sorted(b)) for b in singleBonds]
    
    select_bonds = set(acyclicBonds).intersection(set(singleBonds))
    return select_bonds


class MaxFutureRocsDataset(torch.utils.data.Dataset):
    def __init__(self, database):
        super(MaxFutureRocsDataset, self).__init__()
        self.database = database
        self.database_index = database.index

    def __len__(self):
        return len(self.database)

    def __getitem__(self, key):

        original_mol = deepcopy(self.database.iloc[key].mol_artificial)
        #original_mol = deepcopy(self.database.iloc[key].rdkit_mol_cistrans_stereo)

        dihedral = self.database.iloc[key].dihedral_indices
        positions_before = self.database.iloc[key].positions_before_sorted
        
        if dihedral[3] in positions_before:
            dihedral = tuple(reversed(dihedral))
                
        all_max_rocs = np.zeros(36)
        dihedral_angles = np.zeros(36)
        
        rot_bonds = list(get_acyclic_single_bonds(original_mol))
            
        G = deepcopy(get_substructure_graph_for_matching(original_mol, list(range(0, original_mol.GetNumAtoms()))))
        G.remove_edge(dihedral[1], dihedral[2])
        disjoint_graphs = [list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)]
        
        unknown_positions = sorted(disjoint_graphs[1]) if len(set(positions_before).intersection(set(disjoint_graphs[0]))) > len(set(positions_before).intersection(set(disjoint_graphs[1]))) else sorted(disjoint_graphs[0]) # this should ONLY contain the indices of nodes whose positions are influenced by the dihedral (as well as the focal root node)
        query_and_searched_positions = [u for u in unknown_positions if u not in dihedral[1:3]]
        
        future_rot_bonds = [b for b in rot_bonds if (((b[0] in query_and_searched_positions) | (b[1] in query_and_searched_positions)) & (original_mol.GetAtomWithIdx(b[0]).GetDegree() > 1) & (original_mol.GetAtomWithIdx(b[1]).GetDegree() > 1))]
        
        future_dihedrals = []
        for bond_ in future_rot_bonds:
            if int(bond_[0]) == int(dihedral[2]):
                p1 = ()
            else:
                p1 = rdkit.Chem.rdmolops.GetShortestPath(original_mol, int(bond_[0]), int(dihedral[2]))
            if int(bond_[1]) == int(dihedral[2]):
                p2 = ()
            else:
                p2 = rdkit.Chem.rdmolops.GetShortestPath(original_mol, int(bond_[1]), int(dihedral[2]))
            
            bond = tuple(reversed(bond_)) if len(p1) > len(p2) else bond_ # if bond[0] is farther away, then it should be moved
                
            first_neighbors = tuple(set([a.GetIdx() for a in original_mol.GetAtomWithIdx(bond[0]).GetNeighbors()]) - set([bond[1]]))
            second_neighbors = tuple(set([a.GetIdx() for a in original_mol.GetAtomWithIdx(bond[1]).GetNeighbors()]) - set([bond[0]]))
            d = (first_neighbors[0], *bond, second_neighbors[0])
            future_dihedrals.append(d)
        
        N_confs = max(200*len(future_dihedrals)**1 + 1, 1800) #min(10**len(future_dihedrals), 1000) #32 * 10 * 300
        
        centers_batched_1 = np.zeros((N_confs, len(query_and_searched_positions), 3))
        centers_batched_2 = np.zeros((N_confs, len(query_and_searched_positions), 3))
        alphas_batched_1 = np.ones((N_confs, len(query_and_searched_positions))) * 2.0 #0.81
        alphas_batched_2 = np.ones((N_confs, len(query_and_searched_positions))) * 2.0 #0.81
        prefactors_batched_1 = np.ones((N_confs, len(query_and_searched_positions))) * 0.8
        prefactors_batched_2 = np.ones((N_confs, len(query_and_searched_positions))) * 0.8
        
        random_angles = np.random.uniform(0,360,N_confs*len(future_dihedrals)*36)
        it = 0
        
        for a, angle in enumerate(np.arange(0, 360, 10)):
            
            mol = deepcopy(original_mol)
            
            set_dihedral_angle = float(angle + rdkit.Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *dihedral))
            rdkit.Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(), *dihedral, set_dihedral_angle)
                        
            for i in range(N_confs):
                conf = deepcopy(mol)
                for d in future_dihedrals:
                    rdkit.Chem.rdMolTransforms.SetDihedralDeg(conf.GetConformer(), *d, float(random_angles[it]))
                    it += 1
                                
                centers_batched_1[i] = conf.GetConformer().GetPositions()[query_and_searched_positions]
                centers_batched_2[i] = original_mol.GetConformer().GetPositions()[query_and_searched_positions]
                
            all_rocs_batched = shape_tanimoto_batched(torch.as_tensor(centers_batched_1), torch.as_tensor(centers_batched_2), torch.as_tensor(alphas_batched_1), torch.as_tensor(alphas_batched_2), torch.as_tensor(prefactors_batched_1), torch.as_tensor(prefactors_batched_2)).numpy()
            
            dihedral_angles[a] = set_dihedral_angle
            all_max_rocs[a] = np.max(all_rocs_batched)
            
        return torch.as_tensor(all_max_rocs), torch.as_tensor(dihedral_angles), torch.tensor([self.database_index[key]]).type(torch.long)


# splitting into 16 independent jobs (across 16 nodes of 24 cpu cores each) 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("split", type=int)
args = parser.parse_args()

    
print('reading datasets')

split_idx = int(args.split)
N_splits = 16

filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
artificial_mols_df = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database_artificial_mols.pkl')
filtered_database['mol_artificial'] = artificial_mols_df.artificial_mols

unmerged_future_rocs_db = pd.read_pickle('data/MOSES2/MOSES2_training_val_canonical_terminalSeeds_unmerged_future_rocs_database_all_reduced.pkl').drop_duplicates(['original_index', 'dihedral_indices', 'positions_before_sorted']).reset_index(drop = True)
database = filtered_database[['original_index', 'mol_artificial']].merge(unmerged_future_rocs_db, on='original_index')

splits = [int((len(database) / N_splits) * i) for i in range(N_splits + 1)]
database = database.iloc[ splits[split_idx] : splits[split_idx + 1] ]

dataset = MaxFutureRocsDataset(database)
loader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size = 4, num_workers = 24)

computed_rocs = np.zeros((len(database), 36))
evaluated_dihedrals = np.zeros((len(database), 36))
evaluated_indices = np.zeros(len(database), dtype = int)

if not os.path.exists('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0'):
    os.makedirs('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0')

print('starting workers')

pointer = 0
for b, batch in tqdm(enumerate(loader), total = len(loader)):
    rocs, dihedrals, data_index = batch
    batch_size = rocs.shape[0]

    computed_rocs[pointer:pointer+batch_size] = rocs.numpy()
    evaluated_dihedrals[pointer:pointer+batch_size] = dihedrals.numpy()
    evaluated_indices[pointer:pointer+batch_size] = data_index.squeeze().numpy()

    pointer += batch_size

    if (b+1) % 10000 == 0:
        print(f'saving arrays...batch {b+1}')
        np.save(f'data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_{split_idx}.npy', computed_rocs[0:pointer])
        np.save(f'data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_{split_idx}.npy', evaluated_dihedrals[0:pointer])
        np.save(f'data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_{split_idx}.npy', evaluated_indices[0:pointer])
        print(evaluated_indices)

print(f'saving final arrays...{pointer} entries')
np.save(f'data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_{split_idx}.npy', computed_rocs)
np.save(f'data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_{split_idx}.npy', evaluated_dihedrals)
np.save(f'data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_{split_idx}.npy', evaluated_indices)

print(pointer == len(database))

print('complete.')
