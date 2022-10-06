#import torch_geometric
#import torch
#import torch_scatter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
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

import networkx as nx
import random
from tqdm import tqdm
from multiprocessing import Pool

from utils.general_utils import getNodeFeatures

def get_acyclic_single_bonds(mol):
    AcyclicBonds = rdkit.Chem.MolFromSmarts('[*]!@[*]')
    SingleBonds = rdkit.Chem.MolFromSmarts('[*]-[*]')
    acyclicBonds = mol.GetSubstructMatches(AcyclicBonds)
    singleBonds = mol.GetSubstructMatches(SingleBonds)
    
    acyclicBonds = [tuple(sorted(b)) for b in acyclicBonds]
    singleBonds = [tuple(sorted(b)) for b in singleBonds]
    
    select_bonds = set(acyclicBonds).intersection(set(singleBonds))
    return select_bonds

def get_bond_angle(mol, focal_idx):
    # returns rough bond angles based on atom hybridization
    focal_hybridization = str(mol.GetAtomWithIdx(focal_idx).GetHybridization())
    if focal_hybridization == 'SP':
        return 180.
    elif focal_hybridization == 'SP2':
        return 120.
    elif focal_hybridization == 'SP3':
        return 109.5
    else:
        return None # not implemented for other hybridization states just yet

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

def retrieve_atom_ID(atom_features, atom_lookup):
    atom_ID = [i for i in range(atom_lookup.shape[0]) if np.array_equal(atom_features,atom_lookup[i])][0]
    return atom_ID


def retrieve_bond_ID(bond_prop, bond_lookup_table):
    try:
        bond_ID = int(bond_lookup_table[(bond_lookup_table[0] == bond_prop[0])& \
                                    (bond_lookup_table[1] == bond_prop[1])& \
                                    (bond_lookup_table[2] == bond_prop[2])].index[0])
    except:
        return None
    
    return bond_ID

def fix_bond_angles_and_distance(mol, bond_lookup = None, unique_atoms = None):
    conf = deepcopy(mol)
    
    for p in list(rdkit.Chem.rdmolops.FindAllPathsOfLengthN(conf, 2, useBonds = True)):
        tup = tuple(p)
        bond_1 = conf.GetBondWithIdx(p[0])
        bond_2 = conf.GetBondWithIdx(p[1])
        atom_1, atom_2 = bond_1.GetBeginAtomIdx(), bond_1.GetEndAtomIdx()
        atom_3, atom_4 = bond_2.GetBeginAtomIdx(), bond_2.GetEndAtomIdx()
        
        if (conf.GetAtomWithIdx(atom_1).GetAtomicNum() == 1) | (conf.GetAtomWithIdx(atom_2).GetAtomicNum() == 1) | (conf.GetAtomWithIdx(atom_3).GetAtomicNum() == 1) | (conf.GetAtomWithIdx(atom_4).GetAtomicNum() == 1):
            continue
            
        if atom_1 == atom_3:
            triplet = (atom_2, atom_1, atom_4)
        elif atom_1 == atom_4:
            triplet = (atom_2, atom_1, atom_3)
        elif atom_2 == atom_3:
            triplet = (atom_1, atom_2, atom_4)
        elif atom_2 == atom_4:
            triplet = (atom_1, atom_2, atom_3)
        
        if triplet[0] > triplet[2]:
            triplet = (triplet[2], triplet[1], triplet[0])
        angle = rdkit.Chem.rdMolTransforms.GetAngleDeg(conf.GetConformer(), *triplet)
        
        
        new_angle = get_bond_angle(conf, int(triplet[1]))
        if not conf.GetAtomWithIdx(triplet[1]).IsInRing():
            if np.abs(angle - new_angle) > 1e-4:
                rdkit.Chem.rdMolTransforms.SetAngleDeg(conf.GetConformer(), *triplet, new_angle)
    
    node_features = getNodeFeatures(conf.GetAtoms())
    for bond in conf.GetBonds():
        
        if (bond.GetBeginAtom().GetAtomicNum() == 1) | (bond.GetEndAtom().GetAtomicNum() == 1):
            continue
        
        if (bond.GetBeginAtom().IsInRing() == False) | (bond.GetEndAtom().IsInRing() == False):
            atom1_ID = retrieve_atom_ID(node_features[bond.GetBeginAtomIdx()], unique_atoms[1:])
            atom2_ID = retrieve_atom_ID(node_features[bond.GetEndAtomIdx()], unique_atoms[1:])
            bond_properties = [*sorted([atom1_ID, atom2_ID]), bond.GetBondTypeAsDouble()]
            bond_ID = retrieve_bond_ID(bond_properties, bond_lookup)
            if bond_ID != None:
                bond_distance = bond_lookup.iloc[bond_ID][3]
            else:
                print(f'warning: bond distance between atoms {bond_properties[0]} and {bond_properties[1]} unknown')
                bond_distance = 1.6 # we need a better way of estimating weird bond distances that aren't in the training set
            
            true_bond_distance = rdkit.Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
            rdkit.Chem.rdMolTransforms.SetBondLength(conf.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_distance)
    
    rmse = rdkit.Chem.rdMolAlign.AlignMol(conf, mol)
    
    new_positions = np.array(conf.GetConformer().GetPositions())
    if (True in np.isnan(new_positions)):
        raise Exception('Error in fix_bond_angles_and_distance: Invalid Conformer.')
    
    return conf


def fix_relative_dihedrals(conf):
    rot_bonds = list(get_acyclic_single_bonds(conf))
    reversed_rot_bonds = [(r[1], r[0]) for r in rot_bonds]
    rot_bonds = rot_bonds + reversed_rot_bonds # we want to consider each side of the rotatable bond separately
    
    dihedrals = []
    for bond in rot_bonds:
        
        if conf.GetAtomWithIdx(bond[1]).IsInRing() == True:
            # don't try to adjust coupled dihedrals in a ring structure
            continue
        
        first_neighbors = tuple(set([a.GetIdx() for a in conf.GetAtomWithIdx(bond[0]).GetNeighbors()]) - set([bond[1]]))
        second_neighbors = tuple(set([a.GetIdx() for a in conf.GetAtomWithIdx(bond[1]).GetNeighbors()]) - set([bond[0]]))
        
        if (len(first_neighbors) == 0) | (len(second_neighbors) == 0):
            # not a rotatable bond system (e.g., it includes a terminal atom)
            continue
        
        if len(second_neighbors) < 2:
            # there are no coupled dihedrals to constrain in this case
            continue
            
        d1, d2, d3 = first_neighbors[0], bond[0], bond[1]
        
        d4_anchor = second_neighbors[0] # this neighbor will be locked in place to serve as the reference dihedral
        
        hybrid = str(conf.GetAtomWithIdx(d3).GetHybridization())
        if hybrid == 'SP2':
            rel_angle = 180.
        elif hybrid == 'SP3':
            rel_angle = 120.
        else:
            raise Exception(f'Do not have coupled relative angles for atom {d3} with hybridization {hybrid}')
            
        for d4_query in second_neighbors[1:]:
            
            conf_backup = deepcopy(conf)
            
            bond_angle = rdkit.Chem.rdMolTransforms.GetAngleDeg(conf.GetConformer(), d1, d2, d3)
            if np.abs(bond_angle - 180.) <= 5.:
                # the alignment doesn't work as intended for linear chains like N#N-C-(R)(R)
                continue
            
            conf1 = deepcopy(conf)
    
            d_conf_1 = rdkit.Chem.rdMolTransforms.GetDihedralDeg(conf.GetConformer(), d1, d2, d3, d4_anchor)
            d_conf_2 = rdkit.Chem.rdMolTransforms.GetDihedralDeg(conf.GetConformer(), d1, d2, d3, d4_query)
            dif = (d_conf_1 - d_conf_2 + 180) % 360 - 180
            
            if np.abs(dif - rel_angle) < np.abs(dif - (-rel_angle)):
                add = (rel_angle - dif) % 360
            else:
                add = (-rel_angle - dif) % 360
            
            attempt = d_conf_2 - add
            if np.abs(((d_conf_1 - attempt + 180) % 360 - 180)) - rel_angle > 1e-3:
                attempt = (d_conf_2 + add) % 360
            add = (add + 180) % 360 - 180
            
            rdkit.Chem.rdMolTransforms.SetDihedralDeg(conf1.GetConformer(), d1, d2, d3, d4_query, attempt)
            rmse = rdkit.Chem.rdMolAlign.AlignMol(conf1, conf, atomMap = [(int(at),int(at)) for at in (d1,d2,d3)]) # align to dihedral that isn't being specifically rotated (e.g., the focal rotatable bond)
            
            G = deepcopy(get_substructure_graph_for_matching(conf1, list(range(0, conf1.GetNumAtoms()))))
            G.remove_edge(d3, d4_query)
            disjoint_graphs = [list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)]
            alignment_indices = sorted(disjoint_graphs[0]) if d4_query in disjoint_graphs[0] else sorted(disjoint_graphs[1])
            
            new_positions = conf1.GetConformer().GetPositions()[alignment_indices]
            
            for i,k in enumerate(alignment_indices):
                x,y,z = new_positions[i]
                conf.GetConformer().SetAtomPosition(k, Point3D(x,y,z))
    
    return conf


def fix_bonding_geometries(m):
    m_a = deepcopy(m)
    try:
        m_a = fix_bond_angles_and_distance(m_a, bond_lookup = bond_lookup, unique_atoms = unique_atoms)
        m_a = fix_relative_dihedrals(m_a)
        m_a = fix_bond_angles_and_distance(m_a, bond_lookup = bond_lookup, unique_atoms = unique_atoms)
        fail = 0
    except Exception as e:
        print(e)
        m_a = deepcopy(m)
        fail = 1
        
    return (m_a, fail)


if __name__ == '__main__':

    bond_lookup = pd.read_pickle('data/MOSES2/MOSES2_training_val_bond_lookup.pkl')
    unique_atoms = np.load('data/MOSES2/MOSES2_training_val_unique_atoms.npy')

    filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
    all_mols = list(filtered_database.rdkit_mol_cistrans_stereo)
    
    all_artificial_mols = []
    
    print(f'{len(all_mols)} total mols')
    
    num = 100
    for i in range(num):
        print(i)
        
        mols = deepcopy(all_mols[i*int((len(all_mols) / float(num)) + 1.) : (i+1)*int((len(all_mols) / float(num)) + 1.)])
        print(f'{len(mols)} mols in subset')
        
        artificial_mols = []
        fails = 0
        
        pool = Pool()    
        for i, tup in tqdm(enumerate(pool.imap(fix_bonding_geometries, mols)), total = len(mols)):
            m_a, fail = tup
            artificial_mols.append(m_a)
            fails += fail
        pool.close()
        pool.join()
        
        print(f'{fails} total fails ... {fails/len(mols)}% failure rate.')
        
        all_artificial_mols += artificial_mols
        
    print(f'{len(all_artificial_mols)} total artificial mols')
    
    
    mol_dataframe = pd.DataFrame()
    mol_dataframe['artificial_mols'] = all_artificial_mols
    mol_dataframe.to_pickle('data/MOSES2/MOSES2_training_val_filtered_database_artificial_mols.pkl')
    
    all_xyz = []
    for m_idx, m in tqdm(enumerate(all_artificial_mols), total = len(all_artificial_mols)):
        xyz = m.GetConformer().GetPositions()
        all_xyz.append(xyz)
    xyz_array = np.concatenate(all_xyz, axis = 0)
    np.save('data/MOSES2/MOSES2_training_val_xyz_artificial_array.npy', xyz_array)
