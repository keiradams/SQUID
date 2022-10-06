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

import torch.nn as nn
import torch.nn.functional as F
from models.vnn.models.vn_layers import *
from models.vnn.models.utils.vn_dgcnn_util import get_graph_feature

from utils.general_utils import *
from utils.openeye_utils import *
#from utils.shaep_utils import *

from models.EGNN import *
from models.models import *

import argparse

"""
# example runs:
# python shape_conditioned_generation_evaluations.py lambda10 1.0 0.01
# python shape_conditioned_generation_evaluations.py lambda03 0.3 0.01
"""

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", type=str) # 'lambda10'
parser.add_argument("lambda_interp", type=float) # 1.0
parser.add_argument("stop_threshold", type=float) # 0.01
args = parser.parse_args()


ablateEqui = False


job_id = args.experiment_name
experiment_name = args.experiment_name

stop_threshold = args.stop_threshold
interpolate_to_GNN_prior = args.lambda_interp

variational_GNN_factor = 1.0

if not ablateEqui:
    model_3D_PATH = 'trained_models/graph_generator.pt'
    rocs_model_3D_PATH = 'trained_models/scorer.pt'
else:
    model_3D_PATH = 'trained_models/graph_generator_ablateEqui.pt'
    rocs_model_3D_PATH = 'trained_models/scorer_ablateEqui.pt'

    
PATH = f'shape_conditioned_generation_evaluations_{experiment_name}'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def logger(text, file = PATH + '/logger_' + experiment_name + '.txt', save = True):
    if save:
        with open(file, 'a') as f:
            f.write(text + '\n')
    else:
        print(text + '\n')

logger(f'{experiment_name}')
logger(f'ablate_equi: {ablateEqui}')
logger(f'lambda: {interpolate_to_GNN_prior}')
logger(f'stop_threshold: {stop_threshold}')
logger('')


use_artificial_mols = True

AtomFragment_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl')
AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop = True) # removing stop token from AtomFragment_database

bond_lookup = pd.read_pickle('data/MOSES2/MOSES2_training_val_bond_lookup.pkl')
unique_atoms = np.load('data/MOSES2/MOSES2_training_val_unique_atoms.npy') 

test_mol_df = pd.read_pickle('data/MOSES2/test_MOSES_filtered_artificial_mols.pkl')

if use_artificial_mols:
    test_mols = list(test_mol_df.artificial_mol)
else:
    test_mols = list(test_mol_df.mol)

random.seed(0)
indices_to_evaluate = list(range(0, len(test_mols)))
random.shuffle(indices_to_evaluate)

val_mols = [test_mols[i] for i in indices_to_evaluate]
val_mols_index = indices_to_evaluate


repetitions = 50 
total_evaluations = 1000 


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
    
    ablateEqui = ablateEqui,
    
    old_EGNN = False,
    
).float()

model_3D.load_state_dict(torch.load(model_3D_PATH, map_location=next(model_3D.parameters()).device), strict = True)
model_3D.eval()


# HYPERPARAMETERS for ROCS scorer
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
    
    ablateEqui = ablateEqui,
    
    old_EGNN = False,
    
).float()

rocs_model_3D.load_state_dict(torch.load(rocs_model_3D_PATH, map_location=next(rocs_model_3D.parameters()).device), strict = True)
rocs_model_3D.eval()



# Full 3D generation + scoring
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

seed = 0
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

failed_to_start_rocs = 0
failed_to_get_sequence = 0
failed_to_get_future_sequence = 0

fragment_library_atom_features = np.concatenate(AtomFragment_database['atom_features'], axis = 0).reshape((len(AtomFragment_database), -1))

all_pred_rocs = []
all_random_rocs = []
all_aligned_rocs = []
all_tanimoto = []

all_reference_mols = []
all_aligned_mols = []
all_unaligned_mols = []
all_mols_index = []

all_aligned_rocs_color = []
all_aligned_mols_color = []

for m_idx, m_ in enumerate(val_mols):
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed = seed)
    torch.manual_seed(seed)
    
    m = deepcopy(m_)
    
    mol = deepcopy(m)
    xyz = np.array(mol.GetConformer().GetPositions())
    center_of_mass = np.sum(xyz, axis = 0) / xyz.shape[0]
    xyz_centered = xyz - center_of_mass
    for i in range(0, mol.GetNumAtoms()):
        x,y,z = xyz_centered[i]
        mol.GetConformer().SetAtomPosition(i, Point3D(x,y,z))
    
    m = deepcopy(mol)
    
    
    mol_target = deepcopy(m)
    ring_fragments = get_ring_fragments(mol_target)
    all_possible_seeds = get_all_possible_seeds(mol_target, ring_fragments)
    terminal_seeds = filter_terminal_seeds(all_possible_seeds, mol_target)
    
    select_seeds = []
    for seed in terminal_seeds:
        try:
            frame_generation, frame_rocs = get_frame_terminalSeeds(mol, seed, AtomFragment_database, include_rocs = True)
            positions = list(frame_rocs.iloc[0].positions_before)
            start = 0
            for i in range(len(frame_generation)):
                if (set(frame_generation.iloc[i].partial_graph_indices) == set(positions)) & (frame_generation.iloc[i].next_atom_index == -1):
                    start = i + 1
                    break
            terminalSeed_frame = frame_generation.iloc[0:start].reset_index(drop = True)
            sequence = get_ground_truth_generation_sequence(terminalSeed_frame, AtomFragment_database, fragment_library_atom_features)
            mol = deepcopy(terminalSeed_frame.iloc[0].rdkit_mol_cistrans_stereo)
            partial_indices = deepcopy(terminalSeed_frame.iloc[0].partial_graph_indices_sorted)            
            queue_indices = deepcopy(terminalSeed_frame.iloc[0].focal_indices_sorted)
            _, seed_mol, _, _, _, _, _, _ = generate_seed_from_sequence(sequence, mol, partial_indices, queue_indices, AtomFragment_database, unique_atoms, bond_lookup, stop_after_sequence = True)
            
            if seed_mol.GetNumAtoms() <= 6:
                select_seeds.append(seed)
                continue
        except:
            continue
            
    if len(select_seeds) == 0:
        print('No suitable seed structure. Skipping molecule.')
        continue
    
    random_seed_selection = random.randint(0, len(select_seeds) - 1)
    select_seeds = [select_seeds[random_seed_selection]] * repetitions
    
    
    repeated_rocs = []
    repeated_aligned_rocs = []
    repeated_tanimoto = []
    repeated_aligned_ESP_score = []
    
    repeated_rocs_color = []
    
    reference_mols = []
    aligned_mols = []
    unaligned_mols = []
    
    aligned_mols_color = []
    
    aligned_ESP_mols = []
    
    for seed in select_seeds:
        

        mol = deepcopy(m)
        
        try:
            frame_generation, frame_rocs = get_frame_terminalSeeds(mol, seed, AtomFragment_database, include_rocs = True)
        except Exception as e:
            logger(f'failed to get frame -- {e}')
            continue
        
        if frame_rocs is not None:
            if len(frame_rocs) > 0:
                positions = list(frame_rocs.iloc[0].positions_before)
                start = 0
                for i in range(len(frame_generation)):
                    if (set(frame_generation.iloc[i].partial_graph_indices) == set(positions)) & (frame_generation.iloc[i].next_atom_index == -1):
                        start = i + 1
                        break
            else:
                failed_to_start_rocs += 1
                continue
        else:
            logger('failed to get rocs dataframe')
            failed_to_start_rocs += 1
            continue
        
        if len(frame_generation.iloc[0].partial_graph_indices) == 1: # seed is a terminal ATOM
            terminalSeed_frame = frame_generation.iloc[0:start].reset_index(drop = True)
                    
            try:
                sequence = get_ground_truth_generation_sequence(terminalSeed_frame, AtomFragment_database, fragment_library_atom_features)
            except Exception as e:
                logger(f'failed to get seeding sequence -- {e}')
                failed_to_get_sequence += 1
                continue
                
            mol = deepcopy(terminalSeed_frame.iloc[0].rdkit_mol_cistrans_stereo)
            partial_indices = deepcopy(terminalSeed_frame.iloc[0].partial_graph_indices_sorted)
            
            final_partial_indices = deepcopy(terminalSeed_frame.iloc[-1].partial_graph_indices_sorted)
            ring_fragments = get_ring_fragments(mol)
            add_to_partial = [list(f) for p in final_partial_indices for f in ring_fragments if p in f]
            add_to_partial = [item for sublist in add_to_partial for item in sublist]
            final_partial_indices = list(set(final_partial_indices).union(add_to_partial))
                
            queue_indices = deepcopy(terminalSeed_frame.iloc[0].focal_indices_sorted)
            
            _, seed_mol, queue, positioned_atoms_indices, atom_to_library_ID_map, _, _, _ = generate_seed_from_sequence(sequence, mol, partial_indices, queue_indices, AtomFragment_database, unique_atoms, bond_lookup, stop_after_sequence = True)
    
            seed_node_features = getNodeFeatures(seed_mol.GetAtoms())
            
            for k in atom_to_library_ID_map:
                seed_node_features[k] = AtomFragment_database.iloc[atom_to_library_ID_map[k]].atom_features
                
            G = get_substructure_graph(mol, final_partial_indices)
            G_seed = get_substructure_graph(seed_mol, list(range(0, seed_mol.GetNumAtoms())), node_features = seed_node_features)
            nm = nx.algorithms.isomorphism.generic_node_match(['atom_features'], [None], [np.allclose])
            em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
            GM = nx.algorithms.isomorphism.GraphMatcher(G, G_seed, node_match = nm, edge_match = em)
            assert GM.is_isomorphic()
            idx_map = GM.mapping
                
        else: # seed is a terminal FRAGMENT
            partial_indices = deepcopy(frame_generation.iloc[0].partial_graph_indices_sorted)
            final_partial_indices = partial_indices
            seed_mol = generate_conformer(get_fragment_smiles(mol, partial_indices))
            idx_map = get_reindexing_map(mol, partial_indices, seed_mol)
            positioned_atoms_indices = sorted([idx_map[f] for f in final_partial_indices])
            
            atom_to_library_ID_map = {} # no individual atoms yet generated
            queue = [0] # 0 can be considered the focal root node
    
        assert len(final_partial_indices) == len(seed_mol.GetAtoms())
        for i in final_partial_indices:
            x,y,z = mol.GetConformer().GetPositions()[i]
            seed_mol.GetConformer().SetAtomPosition(idx_map[i], Point3D(x,y,z)) 
        
        starting_queue = deepcopy(queue)
        try:
            _, updated_mol, _, _, _, N_rocs_decisions, _, _, _, chirality_scored = generate_3D_mol_from_sequence(
                sequence = [], 
                partial_mol = deepcopy(seed_mol), 
                mol = deepcopy(mol_target), 
                positioned_atoms_indices = deepcopy(positioned_atoms_indices), 
                queue = starting_queue, 
                atom_to_library_ID_map = deepcopy(atom_to_library_ID_map), 
                model = model_3D, 
                rocs_model = rocs_model_3D,
                AtomFragment_database = AtomFragment_database,
                unique_atoms = unique_atoms, 
                bond_lookup = bond_lookup,
                N_points = N_points_3D, 
                N_points_rocs = N_points_rocs,
                stop_after_sequence = False,
                mask_first_stop = False,
                stochastic = False, 
                chirality_scoring = True,
                stop_threshold = stop_threshold,
                steric_mask = True,
                
                variational_factor_equi = 0.0,
                variational_factor_inv = 0.0, 
                interpolate_to_prior_equi = 0.0,
                interpolate_to_prior_inv = 0.0, 
                
                use_variational_GNN = True, 
                variational_GNN_factor = variational_GNN_factor, 
                interpolate_to_GNN_prior = interpolate_to_GNN_prior, 
                
                rocs_use_variational_GNN = False, 
                rocs_variational_GNN_factor = 0.0, 
                rocs_interpolate_to_GNN_prior = 0.0,
                
                pointCloudVar = pointCloudVar, 
                rocs_pointCloudVar = rocs_pointCloudVar,
            )
            
            pred_rocs = get_ROCS(torch.tensor(np.array(updated_mol.GetConformer().GetPositions())), torch.tensor(np.array(mol.GetConformer().GetPositions())))
            
            tanimoto = rdkit.DataStructs.FingerprintSimilarity(*[rdkit.Chem.RDKFingerprint(x) for x in [mol, updated_mol]])
            
            rocs_shape_score, aligned_rocs, aligned_mol = ROCS_shape_overlap([updated_mol], mol)[0]
            #shaep_mol, shaep_score, aligned_rocs = shape_align(mol, updated_mol, ID = job_id)
            
            reference_mols.append(mol)
            aligned_mols.append(aligned_mol)
            unaligned_mols.append(updated_mol)
            
            repeated_rocs.append(pred_rocs.item())
            repeated_aligned_rocs.append(aligned_rocs)
            repeated_tanimoto.append(tanimoto)
            
            
        except Exception as e:
            logger(f'error during 3D generation -- {e}')
            continue
        
    if len(repeated_rocs) > 0:
        logger(f'{len(all_pred_rocs) + 1}')
        logger(f'mol idx: {val_mols_index[m_idx]}')
        logger(f'{repeated_aligned_rocs}')
        logger(f'{repeated_tanimoto}')
        logger('')
        
        all_pred_rocs.append(repeated_rocs)
        all_aligned_rocs.append(repeated_aligned_rocs)
        all_tanimoto.append(repeated_tanimoto)

        all_reference_mols.append(reference_mols)
        all_aligned_mols.append(aligned_mols)
        all_unaligned_mols.append(unaligned_mols)

        
        all_mols_index.append(val_mols_index[m_idx])
        
        if (len(all_pred_rocs) % 25 == 0):
            logger(f'saving mol objects - {len(all_pred_rocs)}')
            
            with open(f'{PATH}/all_mols_index.pkl', 'wb') as f:
                pickle.dump(all_mols_index, f)
            with open(f'{PATH}/all_reference_mols.pkl', 'wb') as f:
                pickle.dump(all_reference_mols, f)
            with open(f'{PATH}/all_unaligned_mols.pkl', 'wb') as f:
                pickle.dump(all_unaligned_mols, f)
            with open(f'{PATH}/all_aligned_mols.pkl', 'wb') as f:
                pickle.dump(all_aligned_mols, f)
                
    else:
        logger(f'failed to generate mol - {m_idx}')
        
    if len(all_pred_rocs) == total_evaluations:
        break
    
logger('saving final mol objects')
with open(f'{PATH}/all_mols_index.pkl', 'wb') as f:
    pickle.dump(all_mols_index, f)
    
with open(f'{PATH}/all_reference_mols.pkl', 'wb') as f:
    pickle.dump(all_reference_mols, f)
with open(f'{PATH}/all_unaligned_mols.pkl', 'wb') as f:
    pickle.dump(all_unaligned_mols, f)
with open(f'{PATH}/all_aligned_mols.pkl', 'wb') as f:
    pickle.dump(all_aligned_mols, f)


logger('done.')

