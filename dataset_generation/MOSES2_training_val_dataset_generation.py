import torch_geometric
import torch
import torch_scatter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import rdkit.Chem.rdmolops
from rdkit.Chem.rdmolops import AssignStereochemistryFrom3D
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

import networkx as nx
import random
from tqdm import tqdm
import pickle
import os

from utils.general_utils import *

import collections
import collections.abc

from multiprocessing import Pool

def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def logger(text, file = 'MOSES2_training_val_terminalSeeds_generation_reduced_log.txt'):
    with open(file, 'a') as f:
        f.write(text + '\n')
        
        
def conformer_generation(s):
    # s doesn't have stereochemistry specified
    try:
        m_ = rdkit.Chem.MolFromSmiles(s) 
        s_nostereo = rdkit.Chem.MolToSmiles(m_, isomericSmiles = False)
        
        # generate MMFF-optimized conformer and assign stereochemistry from 3D
        m = rdkit.Chem.MolFromSmiles(s_nostereo)
        m = rdkit.Chem.AddHs(m)
        rdkit.Chem.AllChem.EmbedMolecule(m)
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(m, maxIters = 200)
        m = rdkit.Chem.RemoveHs(m)
        m.GetConformer()
        
        n = len(rdkit.Chem.rdmolops.GetMolFrags(m))
        assert n == 1
        
        AssignStereochemistryFrom3D(m)
        
    except:
        m = None
    
    return m


skip = False
if not skip:
    logger('creating MOSES2 mol database...')
    
    smiles_df = pd.read_csv('data/MOSES2/train_MOSES.csv')
    smiles = list(smiles_df.SMILES)
    
    mols = []
    
    pool = Pool()    
    for i, m in tqdm(enumerate(pool.imap(conformer_generation, smiles)), total = len(smiles)):
        if m is not None:
            mols.append(m)
    pool.close()
    pool.join()

    rdkit_smiles = [rdkit.Chem.MolToSmiles(m) for m in mols]
    rdkit_smiles_nostereo = [rdkit.Chem.MolToSmiles(m, isomericSmiles = False) for m in mols]
    
    database = pd.DataFrame()
    database['ID'] = rdkit_smiles
    database['SMILES_nostereo'] = rdkit_smiles_nostereo
    database['rdkit_mol_cistrans_stereo'] = mols
    database['N_atoms'] = [m.GetNumAtoms() for m in mols]
    database['N_rot_bonds'] = [len(get_acyclic_single_bonds(m)) for m in mols]
    
    database = database.drop_duplicates('ID').reset_index(drop = True)
    
    logger(f'database has {len(database)} entries')
    
    logger(f'saving mol database...')
    database.to_pickle('data/MOSES2/MOSES2_training_val_database_mol.pkl')
else:
    database = pd.read_pickle('data/MOSES2/MOSES2_training_val_database_mol.pkl')

bad_confs = []
for m, mol_db in enumerate(database.rdkit_mol_cistrans_stereo):
    try:
        mol_db.GetConformer()
        bad_confs.append(1)
    except:
        bad_confs.append(0)

database['has_conf'] = bad_confs
database = database[database.has_conf == 1].reset_index(drop = True)


def get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment):
    bonds_indices = [b.GetIdx() for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices_sorted = [(b[0], b[1]) if (b[0] in ring_fragment) else (b[1], b[0]) for b in bonded_atom_indices]
    atoms = [b[1] for b in bonded_atom_indices_sorted] 
    return bonds_indices, bonded_atom_indices_sorted, atoms

def get_fragment_smiles(mol, ring_fragment):
    ring_fragment = [int(r) for r in ring_fragment]
    
    try: 
        bonds_indices, bonded_atom_indices_sorted, atoms_bonded_to_ring = get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment)
        pieces = rdkit.Chem.FragmentOnSomeBonds(mol, bonds_indices, numToBreak=len(bonds_indices), addDummies=False) 
    
        fragsMolAtomMapping = []
        fragments = rdkit.Chem.GetMolFrags(pieces[0], asMols = True, sanitizeFrags = True, fragsMolAtomMapping = fragsMolAtomMapping)
        frag_mol = [m_ for i,m_ in enumerate(fragments) if (set(fragsMolAtomMapping[i]) == set(ring_fragment))][0]
    except Exception as e:
        print(f'failed to extract fragment smiles from mol: {rdkit.Chem.MolToSmiles(mol)}, {ring_fragment}')
        print(f'    {e}')
        return None
    
    for a in range(frag_mol.GetNumAtoms()):
        N_rads = frag_mol.GetAtomWithIdx(a).GetNumRadicalElectrons()
        N_Hs = frag_mol.GetAtomWithIdx(a).GetTotalNumHs()
        if N_rads > 0:
            frag_mol.GetAtomWithIdx(a).SetNumExplicitHs(N_rads + N_Hs)
            frag_mol.GetAtomWithIdx(a).SetNumRadicalElectrons(0)
    
    smiles = rdkit.Chem.MolToSmiles(frag_mol, isomericSmiles = False)
    
    smiles_mol = rdkit.Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        print(f'failed to extract fragment smiles: {smiles}, {ring_fragment}')
        return None

    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    return reduced_smiles



skip = False
if not skip:
    logger('creating fragment library... \n')
    
    fragments_smiles = {}
    fragments_smiles_mols = {}
    
    failed = 0
    succeeded = 0
    ignore = set([])
    
    total = len(database.drop_duplicates('ID').rdkit_mol_cistrans_stereo)
    for inc, mol in enumerate(database.drop_duplicates('ID').rdkit_mol_cistrans_stereo):
        
        if inc in [int((total / 20) * j) for j in range(1, 21)]:
            logger(f'    {(inc / total) * 100.} % complete...')
        
        ring_fragments = get_ring_fragments(mol)
        for frag in ring_fragments:

            smiles = get_fragment_smiles(mol, frag)
            
            if not smiles:
                failed += 1
                continue
            else:
                succeeded += 1
            
            if smiles in ignore:
                continue
            
            m = rdkit.Chem.MolFromSmiles(smiles)
            
            # do not include any fragments with radical electrons, for convenience
            N_rads = sum([a.GetNumRadicalElectrons() for a in m.GetAtoms()])
            if N_rads > 0:
                print(f'warning: fragment {smiles} with radical electrons')
                ignore.add(smiles)
                continue
                
            # do not include fragments if we can't generate a conformer for them
            try:
                m_conf = generate_conformer(smiles)
                m_conf.GetConformer()
            except:
                print(f'warning: cannot generate conformer for fragment {smiles}')
                ignore.add(smiles)
                continue

            if smiles not in fragments_smiles:
                fragments_smiles[smiles] = 1
                fragments_smiles_mols[smiles] = rdkit.Chem.MolToSmiles(mol) 
            else:
                fragments_smiles[smiles] += 1
    
    top_k_fragments = {k: v for k, v in sorted(fragments_smiles.items(), key=lambda item: item[1], reverse = True)}
    
    top_100_fragments = {}
    for k in range(0, min(100, len(top_k_fragments))):
        key = list(top_k_fragments.keys())[k]
        top_100_fragments[key] = top_k_fragments[key]
    
    top_100_fragments_smiles = list(top_100_fragments.keys())
    top_100_fragment_library_dict = {s: i for i,s in enumerate(top_100_fragments_smiles)}
    
    # generating optimized fragment geometries
    top_100_fragments_mols = []
    top_100_mols_Hs = []
    for s in top_100_fragments_smiles:
        m_Hs = generate_conformer(s, addHs = True)
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(m_Hs, maxIters = 1000)
        m = rdkit.Chem.RemoveHs(deepcopy(m_Hs))
        top_100_fragments_mols.append(m)
        top_100_mols_Hs.append(m_Hs)
    
    top_100_fragment_database = pd.DataFrame()
    top_100_fragment_database['smiles'] = top_100_fragments_smiles
    top_100_fragment_database['mols'] = top_100_fragments_mols
    top_100_fragment_database['mols_Hs'] = top_100_mols_Hs
    
    logger('saving top 100 fragments...')
    top_100_fragment_database.to_pickle('data/MOSES2/MOSES2_top_100_fragment_database.pkl')

else:
    top_100_fragment_database = pd.read_pickle('data/MOSES2/MOSES2_top_100_fragment_database.pkl')
    top_100_fragments_smiles = list(top_100_fragment_database.smiles)
    top_100_fragments_mols = list(top_100_fragment_database.mols)
    top_100_mols_Hs = list(top_100_fragment_database.mols_Hs)


skip = False
if not skip:
    logger('filtering database... \n')
    
    filtered_database_SMILES_nostereo = []
    m = 0
    SMILES_nostereo_reduced_dataset = database.drop_duplicates('SMILES_nostereo')
    
    total = len(SMILES_nostereo_reduced_dataset)
    for inc, mol in enumerate(SMILES_nostereo_reduced_dataset.rdkit_mol_cistrans_stereo):
        
        if inc in [int((total / 20) * n) for n in range(1, 21)]:
            logger(f'    {(inc / total) * 100.} % complete...')
        
        ring_fragments = get_ring_fragments(mol)
        for frag in ring_fragments:
            smiles = get_fragment_smiles(mol, frag)
            if smiles not in top_100_fragments_smiles:
                break
        else:
            filtered_database_SMILES_nostereo.append(SMILES_nostereo_reduced_dataset.iloc[m].SMILES_nostereo)
        m += 1
    filtered_database_SMILES_nostereo = set(filtered_database_SMILES_nostereo)
    filtered_database_indices = []
    for i in range(len(database)):
        if database.iloc[i].SMILES_nostereo in filtered_database_SMILES_nostereo:
            filtered_database_indices.append(i)
    filtered_database = database.iloc[filtered_database_indices].reset_index(drop = True)
    
    filtered_database['original_index'] = list(range(0, len(filtered_database)))
    
    logger('saving filtered database...')
    filtered_database.to_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
else:
    filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')

    
def get_acyclic_single_bonds(mol):
    AcyclicBonds = rdkit.Chem.MolFromSmarts('[*]!@[*]')
    SingleBonds = rdkit.Chem.MolFromSmarts('[*]-[*]')
    acyclicBonds = mol.GetSubstructMatches(AcyclicBonds)
    singleBonds = mol.GetSubstructMatches(SingleBonds)
    
    acyclicBonds = [tuple(sorted(b)) for b in acyclicBonds]
    singleBonds = [tuple(sorted(b)) for b in singleBonds]
    
    select_bonds = set(acyclicBonds).intersection(set(singleBonds))
    return select_bonds


skip = False
if not skip:
    logger('creating training/val splits...')
    all_smiles = list(set(list(filtered_database.SMILES_nostereo)))
    random.shuffle(all_smiles)
    
    train_smiles = all_smiles[0:int(len(all_smiles)*0.8)]
    val_smiles = all_smiles[int(len(all_smiles)*0.8):]
    
    train_smiles_df = pd.DataFrame()
    train_smiles_df['SMILES_nostereo'] = train_smiles
    train_smiles_df['N_atoms'] = [rdkit.Chem.MolFromSmiles(s).GetNumAtoms() for s in train_smiles]
    train_smiles_df['N_acyclic_single_bonds'] = [len(get_acyclic_single_bonds(rdkit.Chem.MolFromSmiles(s))) for s in train_smiles]
    
    val_smiles_df = pd.DataFrame()
    val_smiles_df['SMILES_nostereo'] = val_smiles
    val_smiles_df['N_atoms'] = [rdkit.Chem.MolFromSmiles(s).GetNumAtoms() for s in val_smiles]
    val_smiles_df['N_acyclic_single_bonds'] = [len(get_acyclic_single_bonds(rdkit.Chem.MolFromSmiles(s))) for s in val_smiles]
    
    train_smiles_df.to_csv('data/MOSES2/MOSES2_train_smiles_split.csv')
    val_smiles_df.to_csv('data/MOSES2/MOSES2_val_smiles_split.csv')


skip = False
if not skip:
    logger('finding unique atoms...')
    unique_atoms_rdkit = []
    total = len(filtered_database.drop_duplicates('ID'))
    for i, mol in enumerate(filtered_database.drop_duplicates('ID')['rdkit_mol_cistrans_stereo']):
        
        if i in [int((total / 20) * n) for n in range(1, 21)]:
            logger(f'    {(i / total) * 100.} % complete...')
        
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = getNodeFeatures(atoms)
        if i == 0:
            unique_atoms = np.zeros((1, node_features.shape[1])) # STOP token
        
        for f, feat in enumerate(node_features):
            N_unique = unique_atoms.shape[0]
            if list(feat) not in unique_atoms.tolist():
                unique_atoms = np.concatenate([unique_atoms, np.expand_dims(feat, axis = 0)], axis = 0) #np.unique(np.concatenate([unique_atoms, np.expand_dims(feat, axis = 0)], axis = 0), axis = 0)
                atom = atoms[f]
                atom.SetChiralTag(rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                unique_atoms_rdkit.append(atom)          
    logger(f'there are {unique_atoms.shape[0]} unique atoms...')
    
    np.save('data/MOSES2/MOSES2_training_val_unique_atoms.npy', unique_atoms)
    
else:
    unique_atoms = np.load('data/MOSES2/MOSES2_training_val_unique_atoms.npy')



skip = False
if not skip:
    logger('creating AtomFragment_database...')
    
    AtomFragment_database = pd.DataFrame()
    AtomFragment_database['mol'] = [None]*unique_atoms.shape[0] + top_100_fragments_mols
    AtomFragment_database['atom_features'] = [ar for ar in np.concatenate((unique_atoms, -1 * np.ones((len(top_100_fragments_mols), unique_atoms.shape[1]))))]
    AtomFragment_database['is_fragment'] = [0]*unique_atoms.shape[0] + [1]*len(top_100_fragments_mols)
    AtomFragment_database['smiles'] = ['']*unique_atoms.shape[0] + top_100_fragments_smiles
    AtomFragment_database['equiv_atoms'] = [list(rdkit.Chem.CanonicalRankAtoms(m, breakTies=False)) if m != None else [0] for m in AtomFragment_database.mol]
    AtomFragment_database['mol_Hs'] = [None]*unique_atoms.shape[0] + top_100_mols_Hs
    
    atom_objects = [None] # stop token
    for atom in unique_atoms_rdkit:
        if atom.GetIsAromatic():
            atom_objects.append(None)
            continue
        n_single = sum([b.GetBondTypeAsDouble() == 1.0 for b in atom.GetBonds()])
        atom_rw = rdkit.Chem.RWMol()
        atom_idx = atom_rw.AddAtom(atom)
        atom_rw = rdkit.Chem.AddHs(atom_rw)
        atom_rw = rdkit.Chem.RWMol(atom_rw)
        for i in range(n_single):
            H = rdkit.Chem.Atom('H')
            H_idx = atom_rw.AddAtom(H)
            atom_rw.AddBond(atom_idx, H_idx)
            atom_rw.GetBondBetweenAtoms(0, i+1).SetBondType(rdkit.Chem.rdchem.BondType.SINGLE)
        atom_rw = rdkit.Chem.RemoveHs(atom_rw)
        atom_rw = rdkit.Chem.AddHs(atom_rw)
        rdkit.Chem.SanitizeMol(atom_rw)
        atom_objects.append(atom_rw)
    AtomFragment_database['atom_objects'] = atom_objects + [None]*len(top_100_fragments_mols)
    
    bond_counts = [translate_node_features(feat)[4:8] for feat in AtomFragment_database.atom_features]
    N_single = [i[0] if AtomFragment_database.iloc[idx].is_fragment == 0 else sum([a.GetTotalNumHs() for a in AtomFragment_database.iloc[idx].mol.GetAtoms()]) for idx, i in enumerate(bond_counts)]
    N_double = [i[1] if AtomFragment_database.iloc[idx].is_fragment == 0 else 0 for idx, i in enumerate(bond_counts)]
    N_triple = [i[2] if AtomFragment_database.iloc[idx].is_fragment == 0 else 0 for idx, i in enumerate(bond_counts)]
    N_aromatic = [i[3] if AtomFragment_database.iloc[idx].is_fragment == 0 else 0 for idx, i in enumerate(bond_counts)]
    AtomFragment_database['N_single'] = N_single
    AtomFragment_database['N_double'] = N_double
    AtomFragment_database['N_triple'] = N_triple
    AtomFragment_database['N_aromatic'] = N_aromatic
    
    AtomFragment_database.to_pickle('data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl')
else:
    AtomFragment_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl')



AtomFragment_database_mols = list(AtomFragment_database.mol)
AtomFragment_database_smiles = np.string_(AtomFragment_database.smiles)
fragment_library_atom_features = np.concatenate(AtomFragment_database['atom_features'], axis = 0).reshape((len(AtomFragment_database), -1))

skip = False
if not skip:
    logger('precomputing mol properties...')
    ## PRECOMPUTE MOL arrays HERE: edge_index, edge_features, node_features, xyz, AND atom_fragment_associations
    all_edge_index = []
    all_edge_features = []
    all_node_features = []
    all_xyz = []
    all_atom_fragment_associations = []
    
    atoms_pointer = np.zeros(len(filtered_database) + 1, dtype = int)
    bonds_pointer = np.zeros(len(filtered_database) + 1, dtype = int)
    a_pointer = 0
    b_pointer = 0
    
    total = len(filtered_database)
    for m_idx, m in enumerate(filtered_database.rdkit_mol_cistrans_stereo):
        
        if m_idx in [int((total / 20.) * n) - 1  for n in range(1, 21)]:
            logger(f'    {((m_idx+1) / total) * 100.} % complete...')
        
        mol = deepcopy(m)
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        edge_index = adjacency_to_undirected_edge_index(adj) # PRECOMPUTE
        
        # Edge Features --> rdkit ordering of edges
        bonds = []
        for b in range(int(edge_index.shape[1]/2)):
            bond_index = edge_index[:,::2][:,b]
            bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
            bonds.append(bond)
        edge_features = getEdgeFeatures(bonds) # PRECOMPUTE
        
        # Node Features --> rdkit ordering of atoms
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = getNodeFeatures(atoms) # PRECOMPUTE
        
        xyz = torch.tensor(mol.GetConformer().GetPositions()) # PRECOMPUTE
        
        ring_fragments = get_ring_fragments(mol)
        atom_fragment_associations = np.zeros(len(atoms), dtype = int)
        for i, atom in enumerate(atoms):
            ring_fragment = [list(r) for r in ring_fragments if i in r]
            if len(ring_fragment) > 0:
                assert len(ring_fragment) == 1
                frag_ID_smiles = get_fragment_smiles(mol, ring_fragment[0])
                atom_fragment_ID_index = np.where(AtomFragment_database_smiles == np.string_(frag_ID_smiles))[0][0]
            else:
                atom_features = node_features[i]
                atom_fragment_ID_index = np.where(np.all(fragment_library_atom_features == atom_features, axis = 1))[0][0]
            atom_fragment_associations[i] = atom_fragment_ID_index
        
        atoms_pointer[m_idx] = a_pointer
        a_pointer += node_features.shape[0]
        
        bonds_pointer[m_idx] = b_pointer
        b_pointer += edge_features.shape[0]
        
        all_edge_index.append(edge_index)
        all_edge_features.append(edge_features)
        all_node_features.append(node_features)
        all_xyz.append(xyz)
        all_atom_fragment_associations.append(atom_fragment_associations)
    
    atoms_pointer[-1] = a_pointer
    bonds_pointer[-1] = b_pointer
        
    edge_index_array = np.concatenate(all_edge_index, axis = 1)
    edge_features_array = np.concatenate(all_edge_features, axis = 0)
    node_features_array = np.concatenate(all_node_features, axis = 0)
    xyz_array = np.concatenate(all_xyz, axis = 0)
    atom_fragment_associations_array = np.concatenate(all_atom_fragment_associations, axis = 0)
    
    logger('saving arrays...')
    np.save('data/MOSES2/MOSES2_training_val_edge_index_array.npy', edge_index_array)
    np.save('data/MOSES2/MOSES2_training_val_edge_features_array.npy', edge_features_array)
    np.save('data/MOSES2/MOSES2_training_val_node_features_array.npy', node_features_array)
    np.save('data/MOSES2/MOSES2_training_val_xyz_array.npy', xyz_array)
    np.save('data/MOSES2/MOSES2_training_val_atom_fragment_associations_array.npy', atom_fragment_associations_array)
    np.save('data/MOSES2/MOSES2_training_val_atoms_pointer.npy', atoms_pointer)
    np.save('data/MOSES2/MOSES2_training_val_bonds_pointer.npy', bonds_pointer)
    
else:
    edge_index_array = np.load('data/MOSES2/MOSES2_training_val_edge_index_array.npy')
    edge_features_array = np.load('data/MOSES2/MOSES2_training_val_edge_features_array.npy')
    node_features_array = np.load('data/MOSES2/MOSES2_training_val_node_features_array.npy')
    xyz_array = np.load('data/MOSES2/MOSES2_training_val_xyz_array.npy')
    atom_fragment_associations_array = np.load('data/MOSES2/MOSES2_training_val_atom_fragment_associations_array.npy')
    atoms_pointer = np.load('data/MOSES2/MOSES2_training_val_atoms_pointer.npy')
    bonds_pointer = np.load('data/MOSES2/MOSES2_training_val_bonds_pointer.npy')

    
skip = False
if not skip:
    logger('creating bond lookup table...')
    
    bond_lookup = pd.DataFrame() # atom_ID, atom_ID, bond_type
    total = len(filtered_database.drop_duplicates('ID'))
    
    for inc, mol_db in enumerate(filtered_database.drop_duplicates('ID')['rdkit_mol_cistrans_stereo']):
        
        if inc in [int((total / 20) * n) for n in range(1, 21)]:
            logger(f'    {(inc / total) * 100.} % complete...')
        
        mol_copy = deepcopy(mol_db)
        rdkit.Chem.RemoveHs(mol_copy)
    
        for bond in mol_copy.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            atom1_ID = retrieve_atom_ID(getNodeFeatures([atom1])[0, :], unique_atoms[1:])
            atom2_ID = retrieve_atom_ID(getNodeFeatures([atom2])[0, :], unique_atoms[1:])
    
            bond_type_double = bond.GetBondTypeAsDouble()
            bond_properties = [*sorted([atom1_ID, atom2_ID]), bond_type_double]
            
            bond_distance = rdkit.Chem.rdMolTransforms.GetBondLength(mol_copy.GetConformer(), int(atom1.GetIdx()), int(atom2.GetIdx()))
            if len(bond_lookup) == 0:
                bond_lookup = bond_lookup.append([[*bond_properties, bond_distance, 1]])
                bond_lookup = bond_lookup.reset_index(drop = True)
                continue
        
            if ((bond_lookup[[0,1,2]] == bond_properties).all(1).any() == False):
                bond_lookup = bond_lookup.append([[*bond_properties, bond_distance, 1]])
                bond_lookup = bond_lookup.reset_index(drop = True)
                
            else:
                bond_ID = retrieve_bond_ID(bond_properties, bond_lookup)
                N = int(bond_lookup.iloc[bond_ID][4]) + 1
                avg = bond_lookup.iloc[bond_ID][3] + ((bond_distance - bond_lookup.iloc[bond_ID][3]) / float(N))
                bond_lookup.iat[bond_ID, 4] = N
                bond_lookup.iat[bond_ID, 3] = avg
                
    bond_lookup.to_pickle('data/MOSES2/MOSES2_training_val_bond_lookup.pkl')
    
else:
    bond_lookup = pd.read_pickle('data/MOSES2/MOSES2_training_val_bond_lookup.pkl')


    
def decompose_from_seed(graph_construction_database, future_rocs_partial_subgraph_database, mol, original_index, seed, canonical):
    
    results = get_partial_subgraphs_BFS(mol, seed[0], canonical = canonical)
                            
    list_2D_partial_graphs, list_2D_focal_atom_fragment, list_2D_focal_attachment, list_2D_next_atom_fragment, list_2D_next_atom_fragment_indices, list_2D_focal_root_node = results[0]
    
    list_positions_before, list_positions_after, list_dihedrals = results[1], results[2], results[3]
    
    for j in range(len(results[0][0])):
        partial_graph_indices = tuple(sorted(list_2D_partial_graphs[j]))
        focal_indices = tuple(sorted(list_2D_focal_atom_fragment[j]))
        focal_attachment_index = list_2D_focal_attachment[j]
        next_atom_index = list_2D_next_atom_fragment[j]
        next_atom_fragment_indices = tuple(sorted(list_2D_next_atom_fragment_indices[j])) if list_2D_next_atom_fragment_indices[j] != -1 else list_2D_next_atom_fragment_indices[j]
        focal_root_node = list_2D_focal_root_node[j]
        seed_tuple = tuple(sorted(seed))
        
        focal_indices_ref_partial = [np.where(np.array(flatten(partial_graph_indices)) == f)[0][0] for f in np.array(flatten(focal_indices))]
        if next_atom_index != -1:
            focal_attachment_index_ref_partial = np.where(np.array(flatten(partial_graph_indices)) == focal_attachment_index)[0][0]
            focal_attachment_point_label_prob = np.zeros(len(focal_indices_ref_partial))
            focal_attachment_point_label_prob[focal_indices_ref_partial.index(focal_attachment_index_ref_partial)] = 1.0
            focal_attachment_point_label_prob = tuple(focal_attachment_point_label_prob)           
        else:
            focal_attachment_index_ref_partial = -1 #tuple([0])
            focal_attachment_point_label_prob = tuple([0.0]*len(focal_indices_ref_partial))
        
        
        if next_atom_index != -1:
            atom_fragment_associations = deepcopy(atom_fragment_associations_array[atoms_pointer[original_index]:atoms_pointer[original_index + 1]])
            
            next_atom_fragment_ID_index = atom_fragment_associations[next_atom_index]
            
            if len(next_atom_fragment_indices) > 1:  
                next_atom_fragment_mol = deepcopy(AtomFragment_database_mols[next_atom_fragment_ID_index])
                next_atom_fragment_attachment_index = get_attachment_index_of_fragment(mol, next_atom_fragment_indices, next_atom_fragment_mol, next_atom_index)
                multi_hot_next_atom_fragment_attachment_points = tuple(get_multi_hot_attachment_points(next_atom_fragment_mol, int(next_atom_fragment_attachment_index)))
            
            else:
                next_atom_fragment_attachment_index = 0
                multi_hot_next_atom_fragment_attachment_points = tuple([1])
            
        else: # STOP token --> need to determine how to include this within batches that aren't only filled with stop tokens
            # not sure what to do here...
            next_atom_fragment_ID_index = 0 # 0 # the first entry in the fragment library is the STOP token
            next_atom_fragment_attachment_index = -1 #torch.tensor(float('nan')).float()
            multi_hot_next_atom_fragment_attachment_points = tuple([-1]) #[None]
            
        # bond type labels of next attachments
        if (next_atom_index != -1): # NOT STOP TOKEN
            attachment_bond_type = str(mol.GetBondBetweenAtoms(int(focal_attachment_index), int(next_atom_index)).GetBondType())
            bond_type_class_index_label = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'].index(attachment_bond_type)
        else:
            bond_type_class_index_label = -1 # -1 
        
        
        graph_construction_database.append([
            original_index, 
            partial_graph_indices, 
            focal_indices, 
            focal_attachment_index, 
            next_atom_index,
            next_atom_fragment_indices,
            focal_root_node,
            seed_tuple,
            focal_attachment_index_ref_partial,
            focal_attachment_point_label_prob,
            next_atom_fragment_attachment_index, # we don't even need this for anything. Also should be ints, not tuples
            multi_hot_next_atom_fragment_attachment_points,
            bond_type_class_index_label,
        ])

        
    for j in range(len(list_positions_before)):
        if (-1) in list_dihedrals[j]:
            continue
        positions_before_sorted = tuple(sorted(list_positions_before[j]))
        positions_after_sorted = tuple(sorted(list_positions_after[j]))
        dihedral_indices = tuple(list_dihedrals[j])
        seed_tuple = tuple(sorted(seed))
        query_indices = tuple(sorted(list(set(positions_after_sorted) - set(positions_before_sorted))))
        query_indices_ref_partial = tuple([positions_after_sorted.index(idx) for idx in query_indices])
        
        future_rocs_partial_subgraph_database.append([
            original_index, 
            positions_before_sorted, 
            positions_after_sorted, 
            dihedral_indices, 
            seed_tuple,
            query_indices,
            query_indices_ref_partial,
        ])
    
    return graph_construction_database, future_rocs_partial_subgraph_database



canonical = True
all_terminal_seeds = True
all_random_seeds = False
only_one_random_seed = False

skip = False
if not skip:
    logger('decomposing molecules... \n')
    
    PATH = 'data/MOSES2/uncombined_databases'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    graph_construction_database = []
    future_rocs_partial_subgraph_database = []
    
    total = len(filtered_database)
    saving_int = 0
    for inc, mol in enumerate(filtered_database.rdkit_mol_cistrans_stereo):
        
        try:
            ring_fragments = get_ring_fragments(mol)
            all_possible_seeds = get_all_possible_seeds(mol, ring_fragments)
            terminal_seeds = filter_terminal_seeds(all_possible_seeds, mol)
            
            if all_terminal_seeds:
                select_seeds = terminal_seeds
            elif all_random_seeds:
                select_seeds = all_possible_seeds
            elif only_one_random_seed:
                select_seeds = [random.choice(all_possible_seeds)]
            
            for seed in select_seeds:
                graph_construction_database, future_rocs_partial_subgraph_database = decompose_from_seed(
                    graph_construction_database, 
                    future_rocs_partial_subgraph_database, 
                    mol = mol, 
                    original_index = inc, 
                    seed = seed, 
                    canonical = canonical,
                )
        except:
            pass
        
        if inc in [int((total / 20.) * n) - 1  for n in range(1, 21)]:
            logger(f'    {((inc+1) / total) * 100.} % complete...')
            
            saving_int += 1
    
            graph_construction_database_pandas = pd.DataFrame()
            graph_construction_database_pandas[['original_index', 'partial_graph_indices_sorted', 'focal_indices_sorted', 'focal_attachment_index', 'next_atom_index', 'next_atom_fragment_indices_sorted', 'focal_root_node', 'seed', 'focal_attachment_index_ref_partial', 'focal_attachment_point_label_prob', 'next_atom_fragment_attachment_index', 'multi_hot_next_atom_fragment_attachment_points', 'bond_type_class_index_label']] = graph_construction_database
            graph_construction_database_pandas['N_atoms_partial'] = [len(indices) for indices in graph_construction_database_pandas.partial_graph_indices_sorted]

            
            future_rocs_partial_subgraph_database_pandas = pd.DataFrame()
            future_rocs_partial_subgraph_database_pandas[['original_index', 'positions_before_sorted', 'positions_after_sorted', 'dihedral_indices', 'seed', 'query_indices', 'query_indices_ref_partial']] = future_rocs_partial_subgraph_database
            future_rocs_partial_subgraph_database_pandas['N_atoms_partial'] = [len(pos) for pos in future_rocs_partial_subgraph_database_pandas.positions_after_sorted]

            graph_construction_database_pandas.to_pickle(f'data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_{saving_int}_reduced.pkl')
            future_rocs_partial_subgraph_database_pandas.to_pickle(f'data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_future_rocs_database_{saving_int}_reduced.pkl')
            
            # re-set for next iteration
            graph_construction_database = []
            future_rocs_partial_subgraph_database = []

