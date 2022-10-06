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

def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


import os
if not os.path.exists('data/MOSES2/training_split_arrays'):
    os.makedirs('data/MOSES2/training_split_arrays')
if not os.path.exists('data/MOSES2/validation_split_arrays'):
    os.makedirs('data/MOSES2/validation_split_arrays')

    

print('creating training arrays...')

filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
unmerged_graph_construction_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_all_reduced.pkl')


train_smiles_df = pd.read_csv('data/MOSES2/MOSES2_train_smiles_split.csv')
train_smiles = set(train_smiles_df.SMILES_nostereo)
train_db_mol = filtered_database.loc[filtered_database['SMILES_nostereo'].isin(train_smiles)].reset_index(drop = True)
train_database = train_db_mol[['original_index', 'N_atoms', 'has_conf', 'rdkit_mol_cistrans_stereo']].merge(unmerged_graph_construction_database, on='original_index')

N_atoms = np.array(train_database.N_atoms)
N_atoms_partial = np.array(train_database.N_atoms_partial)
original_index = np.array(train_database.original_index)
focal_attachment_index = np.array(train_database.focal_attachment_index)
next_atom_index = np.array(train_database.next_atom_index)

partial_graph_indices_sorted = np.array(flatten(train_database.partial_graph_indices_sorted))
partial_graph_indices_sorted_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i,p in tqdm(enumerate(train_database.partial_graph_indices_sorted), total = len(train_database)):
    partial_graph_indices_sorted_pointer[i] = s
    s += len(p)
partial_graph_indices_sorted_pointer[-1] = s

focal_indices_sorted = np.array(flatten(train_database.focal_indices_sorted))
focal_indices_sorted_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.focal_indices_sorted), total = len(train_database)):
    focal_indices_sorted_pointer[i] = s
    s += len(p)
focal_indices_sorted_pointer[-1] = s

next_atom_fragment_indices_sorted = np.array(flatten(train_database.next_atom_fragment_indices_sorted))
next_atom_fragment_indices_sorted_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.next_atom_fragment_indices_sorted), total = len(train_database)):
    next_atom_fragment_indices_sorted_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
next_atom_fragment_indices_sorted_pointer[-1] = s

focal_attachment_index_ref_partial = [f if n != -1 else -1 for f,n in zip(train_database.focal_attachment_index_ref_partial, train_database.next_atom_index)]
train_database['focal_attachment_index_ref_partial'] = focal_attachment_index_ref_partial
focal_attachment_index_ref_partial_array = np.array(train_database.focal_attachment_index_ref_partial)

focal_attachment_point_label_prob_array = np.array(flatten(train_database.focal_attachment_point_label_prob))
focal_attachment_point_label_prob_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.focal_attachment_point_label_prob), total = len(train_database)):
    focal_attachment_point_label_prob_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
focal_attachment_point_label_prob_pointer[-1] = s

multi_hot_next_atom_fragment_attachment_points_array = np.array(flatten(train_database.multi_hot_next_atom_fragment_attachment_points), dtype = int)
multi_hot_next_atom_fragment_attachment_points_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.multi_hot_next_atom_fragment_attachment_points), total = len(train_database)):
    multi_hot_next_atom_fragment_attachment_points_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
multi_hot_next_atom_fragment_attachment_points_pointer[-1] = s

bond_type_class_index_label_array = np.array(train_database.bond_type_class_index_label)

np.save('data/MOSES2/training_split_arrays/N_atoms_partial.npy', N_atoms_partial)
np.save('data/MOSES2/training_split_arrays/N_atoms.npy', N_atoms)
np.save('data/MOSES2/training_split_arrays/original_index.npy', original_index)
np.save('data/MOSES2/training_split_arrays/focal_attachment_index.npy', focal_attachment_index)
np.save('data/MOSES2/training_split_arrays/next_atom_index.npy', next_atom_index)
np.save('data/MOSES2/training_split_arrays/partial_graph_indices_sorted.npy', partial_graph_indices_sorted)
np.save('data/MOSES2/training_split_arrays/partial_graph_indices_sorted_pointer.npy', partial_graph_indices_sorted_pointer)
np.save('data/MOSES2/training_split_arrays/focal_indices_sorted.npy', focal_indices_sorted)
np.save('data/MOSES2/training_split_arrays/focal_indices_sorted_pointer.npy', focal_indices_sorted_pointer)
np.save('data/MOSES2/training_split_arrays/next_atom_fragment_indices_sorted.npy', next_atom_fragment_indices_sorted)
np.save('data/MOSES2/training_split_arrays/next_atom_fragment_indices_sorted_pointer.npy', next_atom_fragment_indices_sorted_pointer)
np.save('data/MOSES2/training_split_arrays/focal_attachment_index_ref_partial_array.npy', focal_attachment_index_ref_partial_array)
np.save('data/MOSES2/training_split_arrays/focal_attachment_point_label_prob_array.npy', focal_attachment_point_label_prob_array)
np.save('data/MOSES2/training_split_arrays/focal_attachment_point_label_prob_pointer.npy', focal_attachment_point_label_prob_pointer)
np.save('data/MOSES2/training_split_arrays/multi_hot_next_atom_fragment_attachment_points_array.npy', multi_hot_next_atom_fragment_attachment_points_array)
np.save('data/MOSES2/training_split_arrays/multi_hot_next_atom_fragment_attachment_points_pointer.npy', multi_hot_next_atom_fragment_attachment_points_pointer)
np.save('data/MOSES2/training_split_arrays/bond_type_class_index_label_array.npy', bond_type_class_index_label_array)




print('creating validation arrays...')

filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
unmerged_graph_construction_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_all_reduced.pkl')

val_smiles_df = pd.read_csv('data/MOSES2/MOSES2_val_smiles_split.csv')
val_smiles = set(val_smiles_df.SMILES_nostereo)
val_db_mol = filtered_database.loc[filtered_database['SMILES_nostereo'].isin(val_smiles)].reset_index(drop = True)
val_database = val_db_mol[['original_index', 'N_atoms', 'has_conf', 'rdkit_mol_cistrans_stereo']].merge(unmerged_graph_construction_database, on='original_index')

N_atoms = np.array(val_database.N_atoms)
N_atoms_partial = np.array(val_database.N_atoms_partial)
original_index = np.array(val_database.original_index)
focal_attachment_index = np.array(val_database.focal_attachment_index)
next_atom_index = np.array(val_database.next_atom_index)

partial_graph_indices_sorted = np.array(flatten(val_database.partial_graph_indices_sorted))
partial_graph_indices_sorted_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i,p in tqdm(enumerate(val_database.partial_graph_indices_sorted), total = len(val_database)):
    partial_graph_indices_sorted_pointer[i] = s
    s += len(p)
partial_graph_indices_sorted_pointer[-1] = s

focal_indices_sorted = np.array(flatten(val_database.focal_indices_sorted))
focal_indices_sorted_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.focal_indices_sorted), total = len(val_database)):
    focal_indices_sorted_pointer[i] = s
    s += len(p)
focal_indices_sorted_pointer[-1] = s

next_atom_fragment_indices_sorted = np.array(flatten(val_database.next_atom_fragment_indices_sorted))
next_atom_fragment_indices_sorted_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.next_atom_fragment_indices_sorted), total = len(val_database)):
    next_atom_fragment_indices_sorted_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
next_atom_fragment_indices_sorted_pointer[-1] = s

focal_attachment_index_ref_partial = [f if n != -1 else -1 for f,n in zip(val_database.focal_attachment_index_ref_partial, val_database.next_atom_index)]
val_database['focal_attachment_index_ref_partial'] = focal_attachment_index_ref_partial
focal_attachment_index_ref_partial_array = np.array(val_database.focal_attachment_index_ref_partial)

focal_attachment_point_label_prob_array = np.array(flatten(val_database.focal_attachment_point_label_prob))
focal_attachment_point_label_prob_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.focal_attachment_point_label_prob), total = len(val_database)):
    focal_attachment_point_label_prob_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
focal_attachment_point_label_prob_pointer[-1] = s

multi_hot_next_atom_fragment_attachment_points_array = np.array(flatten(val_database.multi_hot_next_atom_fragment_attachment_points), dtype = int)
multi_hot_next_atom_fragment_attachment_points_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.multi_hot_next_atom_fragment_attachment_points), total = len(val_database)):
    multi_hot_next_atom_fragment_attachment_points_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
multi_hot_next_atom_fragment_attachment_points_pointer[-1] = s

bond_type_class_index_label_array = np.array(val_database.bond_type_class_index_label)

np.save('data/MOSES2/validation_split_arrays/N_atoms_partial.npy', N_atoms_partial)
np.save('data/MOSES2/validation_split_arrays/N_atoms.npy', N_atoms)
np.save('data/MOSES2/validation_split_arrays/original_index.npy', original_index)
np.save('data/MOSES2/validation_split_arrays/focal_attachment_index.npy', focal_attachment_index)
np.save('data/MOSES2/validation_split_arrays/next_atom_index.npy', next_atom_index)
np.save('data/MOSES2/validation_split_arrays/partial_graph_indices_sorted.npy', partial_graph_indices_sorted)
np.save('data/MOSES2/validation_split_arrays/partial_graph_indices_sorted_pointer.npy', partial_graph_indices_sorted_pointer)
np.save('data/MOSES2/validation_split_arrays/focal_indices_sorted.npy', focal_indices_sorted)
np.save('data/MOSES2/validation_split_arrays/focal_indices_sorted_pointer.npy', focal_indices_sorted_pointer)
np.save('data/MOSES2/validation_split_arrays/next_atom_fragment_indices_sorted.npy', next_atom_fragment_indices_sorted)
np.save('data/MOSES2/validation_split_arrays/next_atom_fragment_indices_sorted_pointer.npy', next_atom_fragment_indices_sorted_pointer)
np.save('data/MOSES2/validation_split_arrays/focal_attachment_index_ref_partial_array.npy', focal_attachment_index_ref_partial_array)
np.save('data/MOSES2/validation_split_arrays/focal_attachment_point_label_prob_array.npy', focal_attachment_point_label_prob_array)
np.save('data/MOSES2/validation_split_arrays/focal_attachment_point_label_prob_pointer.npy', focal_attachment_point_label_prob_pointer)
np.save('data/MOSES2/validation_split_arrays/multi_hot_next_atom_fragment_attachment_points_array.npy', multi_hot_next_atom_fragment_attachment_points_array)
np.save('data/MOSES2/validation_split_arrays/multi_hot_next_atom_fragment_attachment_points_pointer.npy', multi_hot_next_atom_fragment_attachment_points_pointer)
np.save('data/MOSES2/validation_split_arrays/bond_type_class_index_label_array.npy', bond_type_class_index_label_array)

print('done')
