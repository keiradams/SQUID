import torch
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
from rdkit.Geometry import Point3D

import networkx as nx
import random
from tqdm import tqdm
from rdkit.Chem import rdMolTransforms
import itertools
import os
import pickle

import rdkit
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit import Geometry

from utils.openeye_utils import *

# test set target/reference mols
with open(f'paper_results/all_reference_mols.pkl', 'rb') as f:
    all_reference_mols = pickle.load(f)
reference_mols = [deepcopy(all_reference_mols[i][0]) for i in range(len(all_reference_mols))]

# screning training mols
filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
artificial_mols = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database_artificial_mols.pkl')
filtered_database['rdkit_mol_cistrans_stereo'] = artificial_mols.artificial_mols

train_smiles_df = pd.read_csv('data/MOSES2/MOSES2_train_smiles_split.csv')
train_smiles = set(train_smiles_df.SMILES_nostereo)
train_db_mol = filtered_database.loc[filtered_database['SMILES_nostereo'].isin(train_smiles)].reset_index(drop = True)

sreening_mols = list(train_db_mol.rdkit_mol_cistrans_stereo)

best_mols = []

screening_shape_score_results = []
screening_tanimoto_results = []
screening_mol_results = []

for i in tqdm(range(len(reference_mols))):
    
    mol = deepcopy(reference_mols[i])
    
    N_selections = 1000
    n = 0
    
    random_idx = [random.randint(0, len(sreening_mols) - 1) for _ in range(N_selections)]
    
    random_mols = [deepcopy(sreening_mols[r]) for r in random_idx]
    
    screening_rocs_output = ROCS_shape_overlap(random_mols, mol)
    screening_rocs_shape_scores = [r[1] for r in screening_rocs_output]
    screening_tanimoto_similarity = [rdkit.DataStructs.FingerprintSimilarity(*[rdkit.Chem.RDKFingerprint(x) for x in [mol, r_m]]) for r_m in random_mols]
    
    screening_shape_score_results.append(screening_rocs_shape_scores)
    screening_tanimoto_results.append(screening_tanimoto_similarity)
    screening_mol_results.append([r[2] for r in screening_rocs_output])

    
np.save(f'paper_results/dataset_baseline_shape_score_results.npy', np.array(screening_shape_score_results))
np.save(f'paper_results/dataset_baseline_tanimoto_results.npy', np.array(screening_tanimoto_results))
with open(f'paper_results/dataset_baseline_mol_results.pkl', 'wb') as f:
    pickle.dump(screening_mol_results, f)
    
