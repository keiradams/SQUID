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
import networkx as nx
import random
from tqdm import tqdm
from rdkit.Chem import rdMolTransforms
import itertools
import os
import pickle
import shutil

import torch.nn as nn
import torch.nn.functional as F
from models.vnn.models.vn_layers import *
from models.vnn.models.utils.vn_dgcnn_util import get_graph_feature

from utils.general_utils import *
from utils.openeye_utils import *
#from utils.shaep_utils import *

from models.EGNN import *
from models.models import *


"""
# example run:
# python shape_constrained_optimization_evaluations.py GSK3B_99300 GSK3B 99300

# all oracles:
oracle_name_list = ['GSK3B', 
                    'JNK3', 
                    'Osimertinib_MPO', 
                    'Sitagliptin_MPO', 
                    'Celecoxib_Rediscovery', 
                    'Thiothixene_Rediscovery']

#selected 'hit' molecules M_S per oracle:
reference_mol_index_list = [
    [99300, 142337, 94211, 13059, 138951, 67478, 128739, 70016], #GSK3B
    [2775, 7994, 10770, 108203, 126430, 9126, 128739, 70016], #JNK3
    [78600, 81366, 46087, 76561, 87747, 91918, 128739, 70016], #Osimertinib_MPO
    [118822, 132656, 130062, 113584, 115006, 140953, 128739, 70016], #Sitagliptin_MPO
    [33351, 14473, 101938, 6686, 1200, 69153, 128739, 70016], #Celecoxib_Rediscovery
    [25628, 25659, 56430, 137033, 48156, 68289, 128739, 70016], #Thiothixene_Rediscovery
]

"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", type=str) # 'GSK3B_99300'
parser.add_argument("oracle_name", type=str) # GSK3B
parser.add_argument("ref_mol_idx", type=int) # 99300

args = parser.parse_args()
experiment_name = args.experiment_name
oracle_name = args.oracle_name
ref_mol_idx = args.ref_mol_idx

PATH = f'optimization_results/{experiment_name}'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def logger(text, file = PATH + '/logger_' + experiment_name + '.txt', save = True):
    if save:
        with open(file, 'a') as f:
            f.write(text + '\n')
    else:
        print(text + '\n')


model_3D_PATH = 'trained_models/graph_generator.pt'
rocs_model_3D_PATH = 'trained_models/scorer.pt'

job_id = experiment_name

AtomFragment_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl')
AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop = True) # removing stop token from AtomFragment_database!

fragment_library_atom_features = np.concatenate(AtomFragment_database['atom_features'], axis = 0).reshape((len(AtomFragment_database), -1))

bond_lookup = pd.read_pickle('data/MOSES2/MOSES2_training_val_bond_lookup.pkl')
unique_atoms = np.load('data/MOSES2/MOSES2_training_val_unique_atoms.npy')


# HYPERPARAMETERS for 3D graph generator
N_points_3D = 5
pointCloudVar = 1. / (12. * 1.7) 

model_3D = Model_Point_Cloud_Switched(
    input_nf = 45, 
    edges_in_d = 5, 
    n_knn = 5, 
    conv_dims = [32, 32, 64, 128], 
    num_components = 64, 
    fragment_library_dim = 64, 
    N_fragment_layers = 3, 
    append_noise = False, 
    N_members = 125 - 1, 
    EGNN_layer_dim = 64, 
    N_EGNN_layers = 3, 
    output_MLP_hidden_dim = 64, 
    pooling_MLP = False, 
    shared_encoders = False, 
    subtract_latent_space = True,
    variational = False,
    variational_mode = 'inv', # not used
    variational_GNN = True,
    
    mix_node_inv_to_equi = True,
    mix_shape_to_nodes = True,
    ablate_HvarCat = False,
    
    predict_pairwise_properties = False,
    predict_mol_property = False,
    
    ablateEqui = False,
    
    old_EGNN = False,
    
).float()

model_3D.load_state_dict(torch.load(model_3D_PATH, map_location=next(model_3D.parameters()).device), strict = True)
model_3D.eval()


# default HYPERPARAMETERS for ROCS scorer
N_points_rocs = 5
rocs_pointCloudVar = 1. / (12. * 1.7) 

rocs_model_3D = ROCS_Model_Point_Cloud(
    input_nf = 45, 
    edges_in_d = 5, 
    n_knn = 10, 
    conv_dims = [32, 32, 64, 128], 
    num_components = 64, 
    fragment_library_dim = 64,
    N_fragment_layers = 3, 
    append_noise = False, 
    N_members = 125 - 1, 
    EGNN_layer_dim = 64, 
    N_EGNN_layers = 3, 
    output_MLP_hidden_dim = 64, 
    pooling_MLP = False, 
    shared_encoders = False, 
    subtract_latent_space = True,
    variational = False,
    variational_mode = 'inv', # not used
    variational_GNN = False,
    
    mix_node_inv_to_equi = True,
    mix_shape_to_nodes = True,
    ablate_HvarCat = False,
    
    ablateEqui = False,
    
    old_EGNN = False,
    
).float()

rocs_model_3D.load_state_dict(torch.load(rocs_model_3D_PATH, map_location=next(rocs_model_3D.parameters()).device), strict = True)
rocs_model_3D.eval()


######################################

from tdc import Oracle
oracle = Oracle(name = oracle_name) # requires internet access (do)
def score_oracle(mol_list):
    if type(mol_list[0]) is str:
        smiles_list = mol_list
    else:
        smiles_list = [rdkit.Chem.MolToSmiles(m) for m in mol_list]
    scores = oracle(smiles_list)
    return scores


def tanimoto_similarity(mol1, mol2):
    sim = rdkit.DataStructs.FingerprintSimilarity(*[rdkit.Chem.RDKFingerprint(x) for x in [mol1, mol2]])
    return sim

def cross_1to1(H_A, H_B):
    # H_A is (N x d)
    # H_B is (N x d)
    cross_indices = random.sample(list(range(0, H_A.shape[0])),  H_A.shape[0] // 2)
    H_crossed = torch.clone(H_A)
    H_crossed[cross_indices] = torch.clone(H_B)[cross_indices]
    return H_crossed

def mutate(H, interpolate_to_prior = 0.0, sample_std = 1.0):
    H_interp = torch.lerp(H, torch.zeros_like(H), interpolate_to_prior)
    H_mutate = H_interp + sample_std * torch.randn_like(H_interp)
    return H_mutate

######################################

test_mol_df = pd.read_pickle('data/MOSES2/test_MOSES_filtered_artificial_mols.pkl')

test_mols = list(test_mol_df.artificial_mol)
test_indices = list(test_mol_df.index)

reference_mol = deepcopy(test_mols[ref_mol_idx])
reference_score = score_oracle([reference_mol])[0]

logger(experiment_name)
logger(oracle_name)
logger(f'reference mol index (in test split): {test_indices[ref_mol_idx]}')
logger(f'reference mol smiles: {rdkit.Chem.MolToSmiles(reference_mol)}')
logger(f'reference mol oracle score: {reference_score}')

N_initial_mutations_per = 100 
interpolation_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 
stop_threshold = 0.01
N_mutations_per = 10 
N_mates = 10 
N_top_scores = 20 
similarity_cutoff = 0.95
shape_cutoff = 0.75
iterations = 20

S_thresh = 0.845 # target shape similarity constraint (>= 0.85; we round to nearest 0.01)

top_scores_per_iteration = []

mol = deepcopy(reference_mol)

xyz = np.array(mol.GetConformer().GetPositions())
center_of_mass = np.sum(xyz, axis = 0) / xyz.shape[0]
xyz_centered = xyz - center_of_mass
for i in range(0, mol.GetNumAtoms()):
    x,y,z = xyz_centered[i]
    mol.GetConformer().SetAtomPosition(i, Point3D(x,y,z))

ROCS_mol = deepcopy(mol)
    
all_select_seeds = get_starting_seeds(mol, AtomFragment_database, fragment_library_atom_features, unique_atoms, bond_lookup)
assert len(all_select_seeds) > 0
select_seeds = [all_select_seeds[0]]
logger(f'select seeds:{select_seeds}')

mol_history = []
score_history = []
iteration_history = []
population_history = []
shape_scores_history = []

# add starting molecule to population (ensures baseline performance)
_, _, _, _, H_reshaped = encode_molecule_with_generator(
    mol, 
    model_3D, 
    AtomFragment_database, 
    N_points = N_points_3D, 
    pointCloudVar = 1. / (12. * 1.7),
    variational_factor_equi = 0.0, 
    variational_factor_inv = 0.0, 
    interpolate_to_prior_equi = 0.0, 
    interpolate_to_prior_inv = 0.0, 
    use_variational_GNN = True, 
    variational_GNN_factor = 0.0, 
    interpolate_to_GNN_prior = 0.0, 
    h_interpolate = None
)
H = H_reshaped.permute(0,2,1).squeeze(0)

mol_history.append(mol)
score_history.append(reference_score)
iteration_history.append(-1)
shape_scores_history.append(1.0)
population_history.append([H])


# Creating Initial Population by mutating the starting molecule
population = []
sample_std_space = np.ones(N_initial_mutations_per)
for n in range(N_initial_mutations_per):
    for interp in interpolation_values:
        _, _, _, _, H_reshaped = encode_molecule_with_generator(
            mol, 
            model_3D, 
            AtomFragment_database, 
            N_points = N_points_3D, 
            pointCloudVar = 1. / (12. * 1.7),
            variational_factor_equi = 0.0, 
            variational_factor_inv = 0.0, 
            interpolate_to_prior_equi = 0.0, 
            interpolate_to_prior_inv = 0.0, 
            use_variational_GNN = True, 
            variational_GNN_factor = sample_std_space[n], 
            interpolate_to_GNN_prior = interp, 
            h_interpolate = None
        )
        H = H_reshaped.permute(0,2,1).squeeze(0)
        population.append([H])

logger('generating initial population...')
for p in tqdm(population):
    H = p[0]
    gen_mol_list = decode(
        mol, 
        select_seeds, 
        model_3D, 
        rocs_model_3D, 
        AtomFragment_database, 
        fragment_library_atom_features, 
        unique_atoms, 
        bond_lookup, 
        N_points_3D, 
        N_points_rocs, 
        stop_threshold = stop_threshold, 
        variational_GNN_factor = 0.0, 
        interpolate_to_GNN_prior = 0.0, 
        h_interpolate = H, 
                   
        rocs_use_variational_GNN = False, 
        rocs_variational_GNN_factor = 0.0, 
        rocs_interpolate_to_GNN_prior = 0.0, 
        pointCloudVar = 1. / (12. * 1.7), 
        rocs_pointCloudVar = 1. / (12. * 1.7),
    )
    
    if len(gen_mol_list) == 0: continue
    if gen_mol_list[0] is None: continue
    
    s = score_oracle(gen_mol_list)[0]
        
    score_history.append(s)
    mol_history.append(gen_mol_list[0])
    population_history.append(p)
    iteration_history.append(0)

ROCS_output = ROCS_shape_overlap(mol_history, ROCS_mol)
shape_scores_history = [ROCS_output[i][1] for i in range(len(ROCS_output))]
    
for iteration in range(1, iterations+1):
    
    optimization_df = pd.DataFrame()
    optimization_df['iteration_history'] = iteration_history
    optimization_df['mol_history'] = mol_history
    optimization_df['score_history'] = score_history
    optimization_df['shape_scores_history'] = shape_scores_history
    optimization_df['population_history'] = population_history
    optimization_df.to_pickle(PATH + f'/optimization_results_{test_indices[ref_mol_idx]}_incomplete.pkl')
    logger('saved partial results...')
    
    best_score_above_thresh_df = optimization_df[optimization_df.shape_scores_history  >= S_thresh ].score_history
    best_score_above_thresh = max(best_score_above_thresh_df) if len(best_score_above_thresh_df) > 0 else None
    logger(f'top score: {best_score_above_thresh}')
    
    sorted_indices = sorted(range(len(score_history)), key=lambda k: score_history[k], reverse = True)
    
    top_indices = []
    for m_idx in sorted_indices:
        
        if shape_scores_history[m_idx] is None: 
            continue
        
        if shape_scores_history[m_idx] < shape_cutoff:
            continue
        
        skip_m = False
        for t in top_indices:
            if tanimoto_similarity(mol_history[t], mol_history[m_idx]) > similarity_cutoff:
                skip_m = True
                break
                
        if skip_m: continue
        
        top_indices.append(m_idx)
        
        if len(top_indices) == N_top_scores: break
    
    if len(top_indices) == 0:
        logger('population extinct')
        raise Exception('population extinct')
    
    top_H = [population_history[i][0] for i in top_indices]
    top_scores = [score_history[i] for i in top_indices]
    
    
    randomized_top_indices = random.sample(top_indices, len(top_indices))
    random_mates = [tuple(randomized_top_indices[i:i+2]) for i in range(0,N_mates*2,2) if (i+2) < len(randomized_top_indices)]
    for mate in random_mates:
        A, B = mate
        H_A, H_B = population_history[A][0], population_history[B][0]
        H_child = cross_1to1(H_A, H_B)
        top_H.append(H_child)
    
    
    # Creating next population to evaluate
    population = []
    sample_std_space = np.ones(N_mutations_per)
    for H in top_H:
        for n in range(N_mutations_per):
            for interp in interpolation_values:
                H_mutated = mutate(H, interpolate_to_prior = interp, sample_std = sample_std_space[n])
                population.append([H_mutated])
        
    logger(f'generating population for iteration {iteration}...')    
    for p in tqdm(population):
        H = p[0]
        gen_mol_list = decode(
            mol, 
            select_seeds, 
            model_3D, 
            rocs_model_3D, 
            AtomFragment_database, 
            fragment_library_atom_features, 
            unique_atoms, 
            bond_lookup, 
            N_points_3D, 
            N_points_rocs, 
            stop_threshold = stop_threshold, 
            variational_GNN_factor = 0.0, # ignored because we fix h_interpolate
            interpolate_to_GNN_prior = 0.0, # ignored because we fix h_interpolate
            h_interpolate = H, 
                       
            rocs_use_variational_GNN = False, 
            rocs_variational_GNN_factor = 0.0, 
            rocs_interpolate_to_GNN_prior = 0.0, 
            pointCloudVar = 1. / (12. * 1.7), 
            rocs_pointCloudVar = 1. / (12. * 1.7),
        )
        
        if len(gen_mol_list) == 0: continue
        if gen_mol_list[0] is None: continue
        
        s = score_oracle(gen_mol_list)[0]
        
        score_history.append(s)
        mol_history.append(gen_mol_list[0])
        population_history.append(p)
        iteration_history.append(iteration)
    
    ROCS_output = ROCS_shape_overlap(mol_history[len(shape_scores_history): ], ROCS_mol)
    shape_scores_history += [ROCS_output[i][1] for i in range(len(ROCS_output))]

    
sorted_indices = sorted(range(len(score_history)), key=lambda k: score_history[k], reverse = True)

top_indices = []
for m_idx in sorted_indices:
    
    if shape_scores_history[m_idx] is None: 
        continue
    
    if shape_scores_history[m_idx] < shape_cutoff:
        continue
    
    skip_m = False
    for t in top_indices:
        if tanimoto_similarity(mol_history[t], mol_history[m_idx]) > similarity_cutoff:
            skip_m = True
            break
            
    if skip_m: continue
    
    top_indices.append(m_idx)
    
    if len(top_indices) == N_top_scores: break

optimization_df = pd.DataFrame()
optimization_df['iteration_history'] = iteration_history
optimization_df['mol_history'] = mol_history
optimization_df['score_history'] = score_history
optimization_df['shape_scores_history'] = shape_scores_history
optimization_df['population_history'] = population_history

best_score_above_thresh_df = optimization_df[optimization_df.shape_scores_history  >= S_thresh ].score_history
best_score_above_thresh = max(best_score_above_thresh_df) if len(best_score_above_thresh_df) > 0 else None
logger(f'final top score: {best_score_above_thresh}')

optimization_df.to_pickle(PATH + f'/optimization_results_{test_indices[ref_mol_idx]}_complete.pkl')

