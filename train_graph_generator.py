import torch_geometric
import torch

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

import torch_scatter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

import shutil

import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import *
from utils.graph_generator_datasets_and_loaders import *

from models.models import *

import collections
from collections.abc import Mapping, Sequence
from typing import List, Optional, Union
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData

import gc

use_cuda = True
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

save = True

use_artificial_mols = True

mix_node_inv_to_equi = True
mix_shape_to_nodes = True
ablate_HvarCat = False

variational = False # for Z_inv or Z_equi
variational_mode = 'inv' # both, equi, or inv

variational_GNN = True # for GNN-encoded atom embeddings
variational_GNN_mol = False 
cosine_penalty = 0.0

predict_pairwise_properties = False # tanimoto similarity
pairwise_property_factor = 1.0

predict_mol_property = False # specified to be QED in dataset
mol_property_factor = 1.0


shape_penalties = True
shape_penalty_factor = 10.0 if (shape_penalties == True) else 0.0
stop_shape_penalty = 10.0 # 0.0 no penalties

beta_schedule = np.concatenate((np.logspace(-5, -1, 100), np.ones(400)*1e-1))
beta_interval = 10000 # update beta every 10000 iterations (batches)

name = 'training_graph_generator'

PATH = 'results_' + name + '/'
output_file = PATH + name

# change these to load a checkpoint model
model_state = ''
learning_rate_state = None
iteration = 1
beta_iteration = 0 

beta = float(beta_schedule[beta_iteration]) if ((variational == True) | (variational_GNN == True) | (variational_GNN_mol == True)) else None

# Validate only?
validate_only = False

# Data Augmentation
dihedral_var = 15.0
xyz_var = 0.0
randomize_focal_dihedral = True

# HYPERPARAMETERS

ablateEqui = False # switch to True to ablate equivariance

input_nf = 45 
edges_in_d = 5
n_knn = 5
conv_dims = [32, 32, 64, 128]
num_components = 64
fragment_library_dim = 64 
N_fragment_layers = 3
N_members = 125 - 1 
EGNN_layer_dim = 64 
N_EGNN_layers = 3
output_MLP_hidden_dim = 64 

N_points = 5

append_noise = False
learned_noise = False 

pooling_MLP = False
shared_encoders = False
subtract_latent_space = True

target_batch_size = 400

if learning_rate_state is not None:
    lr = learning_rate_state
else:
    lr = 0.00025 

min_lr = 0.00025 / 50. 
use_scheduler = True
gamma = 0.9
num_workers = 20
N_epochs = 100 * 20

# for memory management
chunks = 10
val_chunks = 10

seed = 0
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

if save and (validate_only == False):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        os.makedirs(PATH + 'saved_models/')

    shutil.copyfile('train_graph_generator.py', PATH + 'train_graph_generator.py')
    
def logger(text, file = output_file + '_training_log.txt'):
    if save:
        with open(file, 'a') as f:
            f.write(text + '\n')
    else:
        print(text + '\n')
        
def val_logger(text, file = output_file + '_validation_log.txt'):
    if save | validate_only:
        with open(file, 'a') as f:
            f.write(text + '\n')
    else:
        print(text + '\n')

logger('reading databases')

if (dihedral_var > 0.0) | (xyz_var > 0.0) | (randomize_focal_dihedral == True):
    if use_artificial_mols:
        filtered_database_mols = list(pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database_artificial_mols.pkl').artificial_mols)
    else:
        filtered_database_mols = list(pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database_mols.pkl').rdkit_mol_cistrans_stereo)
else:
    filtered_database_mols = None

logger('reading fragment library')
AtomFragment_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl')
AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop = True) # removing stop token from AtomFragment_database!

AtomFragment_database_mols = list(AtomFragment_database.mol)
AtomFragment_database_smiles = list(AtomFragment_database.smiles)
fragment_library_atom_features = np.concatenate(AtomFragment_database['atom_features'], axis = 0).reshape((len(AtomFragment_database), -1))

logger('computing fragment shape penalties')
if shape_penalties:
    volumes = np.zeros(len(AtomFragment_database))
    for i in range(len(AtomFragment_database)):
        m = deepcopy(AtomFragment_database.iloc[i].mol)
        a = deepcopy(AtomFragment_database.iloc[i].atom_objects)
        if m is not None:
            volumes[i] = (rdkit.Chem.AllChem.ComputeMolVolume(m))
        elif a is not None:
            rdkit.Chem.AllChem.EmbedMolecule(a)
            a = rdkit.Chem.RemoveHs(a)
            volumes[i] = (rdkit.Chem.AllChem.ComputeMolVolume(a))
        else:
            volumes[i] = 0.0
    volume_distances = torch.cdist(torch.as_tensor(volumes).unsqueeze(1), torch.as_tensor(volumes).unsqueeze(1)).float()
    volume_distances = volume_distances.to(device)
    
else:
    volume_distances = torch.zeros((len(AtomFragment_database), len(AtomFragment_database))).float()

logger('reading common data arrays')
    
# these are common to both training and validation splits.
edge_index_array = np.load('data/MOSES2/MOSES2_training_val_edge_index_array.npy')
edge_features_array = np.load('data/MOSES2/MOSES2_training_val_edge_features_array.npy')
node_features_array = np.load('data/MOSES2/MOSES2_training_val_node_features_array.npy')

if use_artificial_mols:
    xyz_array = np.load('data/MOSES2/MOSES2_training_val_xyz_artificial_array.npy')
else:
    xyz_array = np.load('data/MOSES2/MOSES2_training_val_xyz_array.npy')

atom_fragment_associations_array = np.load('data/MOSES2/MOSES2_training_val_atom_fragment_associations_array.npy')
atom_fragment_associations_array = atom_fragment_associations_array - 1

atoms_pointer = np.load('data/MOSES2/MOSES2_training_val_atoms_pointer.npy')
bonds_pointer = np.load('data/MOSES2/MOSES2_training_val_bonds_pointer.npy')


logger('reading training data arrays')

original_index = np.load('data/MOSES2/training_split_arrays/original_index.npy')
N_atoms_partial = np.load('data/MOSES2/training_split_arrays/N_atoms_partial.npy')
N_atoms = np.load('data/MOSES2/training_split_arrays/N_atoms.npy')
focal_attachment_index = np.load('data/MOSES2/training_split_arrays/focal_attachment_index.npy')
next_atom_index = np.load('data/MOSES2/training_split_arrays/next_atom_index.npy')

partial_graph_indices_sorted = np.load('data/MOSES2/training_split_arrays/partial_graph_indices_sorted.npy')
partial_graph_indices_sorted_pointer = np.load('data/MOSES2/training_split_arrays/partial_graph_indices_sorted_pointer.npy')

focal_indices_sorted = np.load('data/MOSES2/training_split_arrays/focal_indices_sorted.npy')
focal_indices_sorted_pointer = np.load('data/MOSES2/training_split_arrays/focal_indices_sorted_pointer.npy')

next_atom_fragment_indices_sorted = np.load('data/MOSES2/training_split_arrays/next_atom_fragment_indices_sorted.npy')
next_atom_fragment_indices_sorted_pointer = np.load('data/MOSES2/training_split_arrays/next_atom_fragment_indices_sorted_pointer.npy')

focal_attachment_index_ref_partial_array = np.load('data/MOSES2/training_split_arrays/focal_attachment_index_ref_partial_array.npy')
focal_attachment_point_label_prob_array = np.load('data/MOSES2/training_split_arrays/focal_attachment_point_label_prob_array.npy')
focal_attachment_point_label_prob_pointer = np.load('data/MOSES2/training_split_arrays/focal_attachment_point_label_prob_pointer.npy')

multi_hot_next_atom_fragment_attachment_points_array = np.load('data/MOSES2/training_split_arrays/multi_hot_next_atom_fragment_attachment_points_array.npy')
multi_hot_next_atom_fragment_attachment_points_pointer = np.load('data/MOSES2/training_split_arrays/multi_hot_next_atom_fragment_attachment_points_pointer.npy')

bond_type_class_index_label_array = np.load('data/MOSES2/training_split_arrays/bond_type_class_index_label_array.npy')


logger('reading validation data arrays')

val_original_index = np.load('data/MOSES2/validation_split_arrays/original_index.npy')
val_N_atoms_partial = np.load('data/MOSES2/validation_split_arrays/N_atoms_partial.npy')
val_N_atoms = np.load('data/MOSES2/validation_split_arrays/N_atoms.npy')
val_focal_attachment_index = np.load('data/MOSES2/validation_split_arrays/focal_attachment_index.npy')
val_next_atom_index = np.load('data/MOSES2/validation_split_arrays/next_atom_index.npy')

val_partial_graph_indices_sorted = np.load('data/MOSES2/validation_split_arrays/partial_graph_indices_sorted.npy')
val_partial_graph_indices_sorted_pointer = np.load('data/MOSES2/validation_split_arrays/partial_graph_indices_sorted_pointer.npy')

val_focal_indices_sorted = np.load('data/MOSES2/validation_split_arrays/focal_indices_sorted.npy')
val_focal_indices_sorted_pointer = np.load('data/MOSES2/validation_split_arrays/focal_indices_sorted_pointer.npy')

val_next_atom_fragment_indices_sorted = np.load('data/MOSES2/validation_split_arrays/next_atom_fragment_indices_sorted.npy')
val_next_atom_fragment_indices_sorted_pointer = np.load('data/MOSES2/validation_split_arrays/next_atom_fragment_indices_sorted_pointer.npy')

val_focal_attachment_index_ref_partial_array = np.load('data/MOSES2/validation_split_arrays/focal_attachment_index_ref_partial_array.npy')
val_focal_attachment_point_label_prob_array = np.load('data/MOSES2/validation_split_arrays/focal_attachment_point_label_prob_array.npy')
val_focal_attachment_point_label_prob_pointer = np.load('data/MOSES2/validation_split_arrays/focal_attachment_point_label_prob_pointer.npy')

val_multi_hot_next_atom_fragment_attachment_points_array = np.load('data/MOSES2/validation_split_arrays/multi_hot_next_atom_fragment_attachment_points_array.npy')
val_multi_hot_next_atom_fragment_attachment_points_pointer = np.load('data/MOSES2/validation_split_arrays/multi_hot_next_atom_fragment_attachment_points_pointer.npy')

val_bond_type_class_index_label_array = np.load('data/MOSES2/validation_split_arrays/bond_type_class_index_label_array.npy')


logger('initializing model')
model = Model_Point_Cloud_Switched(
    input_nf = input_nf, 
    edges_in_d = edges_in_d, 
    n_knn = n_knn, 
    conv_dims = conv_dims, 
    num_components = num_components, 
    fragment_library_dim = fragment_library_dim, 
    N_fragment_layers = N_fragment_layers, 
    append_noise = append_noise, 
    N_members = N_members, 
    EGNN_layer_dim = EGNN_layer_dim, 
    N_EGNN_layers = N_EGNN_layers, 
    output_MLP_hidden_dim = output_MLP_hidden_dim, 
    pooling_MLP = pooling_MLP, 
    shared_encoders = shared_encoders, 
    subtract_latent_space = subtract_latent_space,
    variational = variational,
    variational_mode = variational_mode,
    variational_GNN = variational_GNN,
    variational_GNN_mol = variational_GNN_mol,
    
    mix_node_inv_to_equi = mix_node_inv_to_equi,
    mix_shape_to_nodes = mix_shape_to_nodes,
    ablate_HvarCat = ablate_HvarCat,
    
    predict_pairwise_properties = predict_pairwise_properties,
    predict_mol_property = predict_mol_property,
    
    ablateEqui = ablateEqui,
    
    old_EGNN = False,
    
).float()

if (model.append_noise == True) and (learned_noise == False):
    for p in model.Encoder.fragment_encoder.noise_embedding.parameters():
        p.requires_grad = False

if model_state != '':
    logger(f'loading model parameters from {model_state}')
    model.load_state_dict(torch.load(model_state, map_location=next(model.parameters()).device), strict=True)

model.to(device)
        
logger(f'model has {sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])} parameters')


logger('creating datasets and dataloaders')

library_dataset = AtomFragmentLibrary(AtomFragment_database)
library_loader = torch_geometric.data.DataLoader(
    library_dataset, 
    shuffle = False, 
    batch_size = len(library_dataset), 
    num_workers = 0,
)
fragment_batch = next(iter(library_loader))
N_fragment_library_nodes = fragment_batch.x.shape[0]
fragment_batch_batch = fragment_batch.batch
fragment_batch = fragment_batch.to(device)


sampler_df = pd.DataFrame()
sampler_df['N_atoms'] = N_atoms
sampler_df['N_atoms_partial'] = N_atoms_partial

train_sampler = VNNBatchSampler(sampler_df, target_batch_size, chunks = chunks)
train_dataset = FragmentGraphDataset_point_cloud(

    filtered_database_mols,

    original_index,

    edge_index_array, 
    edge_features_array, 
    node_features_array, 
    xyz_array, 
    atom_fragment_associations_array, 
    atoms_pointer, 
    bonds_pointer,

    focal_attachment_index,
    next_atom_index,
    partial_graph_indices_sorted,
    partial_graph_indices_sorted_pointer,
    focal_indices_sorted,
    focal_indices_sorted_pointer,
    next_atom_fragment_indices_sorted,
    next_atom_fragment_indices_sorted_pointer,

    focal_attachment_index_ref_partial_array,
    focal_attachment_point_label_prob_array,
    focal_attachment_point_label_prob_pointer,
    multi_hot_next_atom_fragment_attachment_points_array,
    multi_hot_next_atom_fragment_attachment_points_pointer,
    bond_type_class_index_label_array,

    N_points = N_points,

    dihedral_var = dihedral_var,
    xyz_var = xyz_var,
    randomize_focal_dihedral = randomize_focal_dihedral,

)

train_loader = DataLoader(
    train_dataset, 
    batch_sampler = train_sampler,
    num_workers = num_workers,
    persistent_workers = False, 
    follow_batch = ['x', 'x_subgraph'], 
    N_fragment_library_nodes = N_fragment_library_nodes, 
    fragment_batch_batch = fragment_batch_batch,
    )


val_sampler_df = pd.DataFrame()
val_sampler_df['N_atoms'] = val_N_atoms
val_sampler_df['N_atoms_partial'] = val_N_atoms_partial

val_sampler = VNNBatchSampler(val_sampler_df, target_batch_size, chunks = val_chunks)
val_dataset = FragmentGraphDataset_point_cloud(

    filtered_database_mols,

    val_original_index,

    edge_index_array, 
    edge_features_array, 
    node_features_array, 
    xyz_array, 
    atom_fragment_associations_array, 
    atoms_pointer, 
    bonds_pointer,

    val_focal_attachment_index,
    val_next_atom_index,
    val_partial_graph_indices_sorted,
    val_partial_graph_indices_sorted_pointer,
    val_focal_indices_sorted,
    val_focal_indices_sorted_pointer,
    val_next_atom_fragment_indices_sorted,
    val_next_atom_fragment_indices_sorted_pointer,

    val_focal_attachment_index_ref_partial_array,
    val_focal_attachment_point_label_prob_array,
    val_focal_attachment_point_label_prob_pointer,
    val_multi_hot_next_atom_fragment_attachment_points_array,
    val_multi_hot_next_atom_fragment_attachment_points_pointer,
    val_bond_type_class_index_label_array,

    N_points = N_points,

    dihedral_var = dihedral_var,
    xyz_var = xyz_var,
    randomize_focal_dihedral = randomize_focal_dihedral,
)

val_loader = DataLoader(
    val_dataset, 
    batch_sampler = val_sampler,
    num_workers = num_workers,
    persistent_workers = False,
    follow_batch = ['x', 'x_subgraph'], 
    N_fragment_library_nodes = N_fragment_library_nodes, 
    fragment_batch_batch = fragment_batch_batch,
    )


def loop(model, optimizer, batch_data, training = True, device = torch.device('cpu'), shape_penalty_factor = 0.0, volume_distances = None, stop_shape_penalty = 0.0, variational = False, variational_mode = 'both', variational_GNN = False, variational_GNN_mol = False, cosine_penalty = 0.0, beta = 0.0, predict_pairwise_properties = False, pairwise_property_factor = 1.0, predict_mol_property = False, mol_property_factor = 1.0):

    batch, batch_dict = deepcopy(batch_data)

    batch_size = batch_dict['batch_size']
    
    if batch_size == 1:
        return 1, torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), 0, 0, 0, 0, torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), 0, 0, torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), torch.tensor(float('NaN')).item(), 0, torch.tensor(float('NaN'))
        
    if training:
        optimizer.zero_grad()

    focal_index_rel_partial = batch_dict['focal_index_rel_partial']
    focal_indices_batch = batch_dict['focal_indices_batch'] 
    stop_mask = batch_dict['stop_mask']
    stop_focal_mask = batch_dict['stop_focal_mask']
    all_stop = batch_dict['all_stop']
    next_atomFragment_attachment_loss_mask_size = batch_dict['next_atomFragment_attachment_loss_mask_size'] 
    focal_attachment_loss_mask = batch_dict['focal_attachment_loss_mask']
    focal_attachment_loss_mask_size = batch_dict['focal_attachment_loss_mask_size']
    select_multi_losses = batch_dict['select_multi_losses']
    mask_select_multi = batch_dict['mask_select_multi']
    focal_attachment_label_prob_masked = batch_dict['focal_attachment_label_prob_masked']
    masked_focal_batch_index_reindexed = batch_dict['masked_focal_batch_index_reindexed']
    focal_attachment_index_rel_partial = batch_dict['focal_attachment_index_rel_partial']
    next_atom_attachment_indices = batch_dict['next_atom_attachment_indices']
    masked_next_atom_attachment_batch_index_reindexed = batch_dict['masked_next_atom_attachment_batch_index_reindexed']
    masked_multihot_next_attachments = batch_dict['masked_multihot_next_attachments'] 
    masked_multihot_next_attachments_label_prob = batch_dict['masked_multihot_next_attachments_label_prob']
    bond_type_mask = batch_dict['bond_type_mask']
    
    pairwise_indices_1_select = batch_dict['pairwise_indices_1_select']
    pairwise_indices_2_select = batch_dict['pairwise_indices_2_select']
    pairwise_targets = batch_dict['pairwise_targets']
    
    pairwise_indices_1_select = pairwise_indices_1_select.to(device)
    pairwise_indices_2_select = pairwise_indices_2_select.to(device)
    pairwise_targets = pairwise_targets.to(device)
    
    batch = batch.to(device)
    
    mol_prop_targets = batch.mol_prop.float()
    
    focal_index_rel_partial = focal_index_rel_partial.to(device)
    focal_indices_batch = focal_indices_batch.to(device)

    stop_mask = stop_mask.to(device)
    stop_focal_mask = stop_focal_mask.to(device)

    if not all_stop:
        focal_attachment_loss_mask = focal_attachment_loss_mask.to(device)
        focal_attachment_label_prob_masked = focal_attachment_label_prob_masked.to(device)
        masked_focal_batch_index_reindexed = masked_focal_batch_index_reindexed.to(device)
        focal_attachment_index_rel_partial = focal_attachment_index_rel_partial.to(device) 
        next_atom_attachment_indices = next_atom_attachment_indices.to(device) 
        masked_next_atom_attachment_batch_index_reindexed = masked_next_atom_attachment_batch_index_reindexed.to(device)
        masked_multihot_next_attachments = masked_multihot_next_attachments.to(device)
        bond_type_mask = bond_type_mask.to(device) # to(device) might not be needed here...

        if next_atomFragment_attachment_loss_mask_size > 0:
            select_multi_losses = select_multi_losses.to(device)
            mask_select_multi = mask_select_multi.to(device)

    args = (
        batch_size, 
        
        batch.x.float(), 
        batch.edge_index, 
        batch.edge_attr.float(), 
        batch.pos.float(), 
        batch.cloud.float(), 
        batch.cloud_indices, 
        batch.atom_fragment_associations, 
        
        batch.x_subgraph.float(), 
        batch.edge_index_subgraph, 
        batch.edge_attr_subgraph.float(), 
        batch.pos_subgraph.float(),
        batch.cloud_subgraph.float(),
        batch.cloud_indices_subgraph,
        batch.atom_fragment_associations_subgraph,
    
        focal_index_rel_partial, 
        focal_indices_batch, 
        
        fragment_batch, 
        batch.next_atom_type_library_idx,
        stop_mask, 
        stop_focal_mask, 
        masked_focal_batch_index_reindexed, 
        focal_attachment_index_rel_partial, 
        next_atom_attachment_indices, 
        masked_next_atom_attachment_batch_index_reindexed, 
        masked_multihot_next_attachments, 
    )
    
    if not training:
        with torch.no_grad():
            out = model(*args, 
                        all_stop = all_stop, 
                        pairwise_indices_1_select = pairwise_indices_1_select,
                        pairwise_indices_2_select = pairwise_indices_2_select,
                        device = device)
    else:
        out = model(*args, 
                    all_stop = all_stop,
                    pairwise_indices_1_select = pairwise_indices_1_select,
                    pairwise_indices_2_select = pairwise_indices_2_select,
                    device = device)
    
    # predicting stop tokens
    stop_loss = stop_BCE(torch.sigmoid(out[0].squeeze()), stop_mask.type(torch.float))
    stop_accuracy = torch.mean((torch.round(torch.sigmoid(out[0].squeeze())) == stop_mask.type(torch.float)).type(torch.float))
    backprop_loss = stop_loss
    
    # stop shape loss
    if not all_stop:
        N_future_atoms = batch.N_future_atoms.squeeze()
        stop_shape_loss = torch.mean((1.0 - torch.sigmoid(out[0].squeeze()[stop_mask])) * batch.N_future_atoms.squeeze()[stop_mask])
        backprop_loss = backprop_loss + stop_shape_loss * stop_shape_penalty
    else:
        stop_shape_loss = torch.tensor(float('NaN')) # Nan
    
    
    if variational:
        Z_equi_mean, Z_equi_std, Z_inv_mean, Z_inv_std = out[5], out[6], out[7], out[8]
        
        if (variational_mode == 'both') | (variational_mode == 'equi'):
            KL_unreduced_equi = 0.5 * (torch.sum(Z_equi_mean.reshape(batch_size, -1)**2.0, dim = 1) + torch.sum(Z_equi_std.reshape(batch_size, -1)**2.0, dim = 1) - torch.sum(torch.log(Z_equi_std.reshape(batch_size, -1)**2.0) + 1.0, dim = 1))
        if (variational_mode == 'both') | (variational_mode == 'inv'):
            KL_unreduced_inv = 0.5 * (torch.sum(Z_inv_mean**2.0, dim = 1) + torch.sum(Z_inv_std**2.0, dim = 1) - torch.sum(torch.log(Z_inv_std**2.0) + 1.0, dim = 1))
        
        if variational_mode == 'both':
            KL_loss = torch.mean(KL_unreduced_equi) + torch.mean(KL_unreduced_inv)
        elif variational_mode == 'inv':
            KL_loss = torch.mean(KL_unreduced_inv)
        elif variational_mode == 'equi':
            KL_loss = torch.mean(KL_unreduced_equi)
            
        backprop_loss = backprop_loss + beta * KL_loss
        
        cosine_loss = torch.tensor(float('NaN')) # Nan 
    
    elif variational_GNN: # since batches contain molecules with same # of atoms, we don't need to do any additional averaging
        h_mean, h_std = out[9], out[10]
        KL_unreduced = 0.5 * (torch.sum(h_mean**2.0, dim = 1) + torch.sum(h_std**2.0, dim = 1) - torch.sum(torch.log(h_std**2.0) + 1.0, dim = 1))
        KL_loss = torch.mean(KL_unreduced)
        
        backprop_loss = backprop_loss + beta * KL_loss
        
        cosine_loss = torch.tensor(float('NaN')) # Nan 
        
    elif variational_GNN_mol:
        h_mean, h_std = out[9], out[10]
        KL_unreduced = 0.5 * (torch.sum(h_mean**2.0, dim = 1) + torch.sum(h_std**2.0, dim = 1) - torch.sum(torch.log(h_std**2.0) + 1.0, dim = 1))
        KL_loss = torch.mean(KL_unreduced)
        
        backprop_loss = backprop_loss + beta * KL_loss
        
        h_reshaped_gnn, h_predicted_reshaped = out[11], out[12]
        cosine_loss = torch.mean(1. - torch.nn.functional.cosine_similarity(h_reshaped_gnn, h_predicted_reshaped, dim=1))
        
        backprop_loss = backprop_loss + cosine_penalty * cosine_loss
        
        
    else: # no variational components in encoder
        KL_loss = torch.tensor(float('NaN')) # Nan
        cosine_loss = torch.tensor(float('NaN')) # Nan
        
    
    if predict_pairwise_properties:
        pairwise_properties_out = out[14]
        pairwise_properties_out_sigmoid = torch.sigmoid(pairwise_properties_out.squeeze())
        pairwise_property_loss = torch.mean(torch.square(pairwise_targets - pairwise_properties_out_sigmoid))
        N_pairwise_targets = pairwise_targets.shape[0]
        
        backprop_loss = backprop_loss + pairwise_property_factor*pairwise_property_loss
    else:
        pairwise_property_loss = torch.tensor(float('NaN')) # Nan
        N_pairwise_targets = 0
    
    
    if predict_mol_property:
        mol_prop_out = out[15]
        mol_prop_out_sigmoid = torch.sigmoid(mol_prop_out.squeeze())
        mol_prop_loss = torch.mean(torch.square(mol_prop_targets - mol_prop_out_sigmoid))
        
        backprop_loss = backprop_loss + mol_property_factor * mol_prop_loss
    else:
        mol_prop_loss = torch.tensor(float('NaN')) # Nan
        
        
    if not all_stop:

        # predicting next atom/fragment types (only for non-stop tokens)
        next_atomFragment_loss = next_atomFragment_cross_entropy(out[1], batch.next_atom_type_library_idx[stop_mask])
        next_atomFragment_accuracy = torch.mean((out[1].softmax(dim = 1).argmax(dim = 1) == batch.next_atom_type_library_idx[stop_mask]).type(torch.float))
        backprop_loss = backprop_loss + next_atomFragment_loss

        # adding in shape penalty
        if shape_penalty_factor > 0.0:
            shape_loss = torch.mean(out[1].softmax(dim = 1) * volume_distances[batch.next_atom_type_library_idx[stop_mask]])
            backprop_loss = backprop_loss + shape_penalty_factor*shape_loss
        else:
            shape_loss = torch.tensor(float('NaN')) # Nan

    
        # predicing focal fragment attachment point
        focal_attachment_losses = torch_scatter.scatter_sum(-torch.log(out[2] + 1e-8)*focal_attachment_label_prob_masked, masked_focal_batch_index_reindexed, dim = 0) 

        if focal_attachment_loss_mask_size > 0:
            focal_attachment_loss = torch.mean(focal_attachment_losses[focal_attachment_loss_mask]) # selecting only those losses from actual fragments (I think)
            backprop_loss = backprop_loss + focal_attachment_loss
        else:
            focal_attachment_loss = torch.tensor(float('NaN'))
        
        # predicting next atom/fragment attachments with graph-equivalency and masking non-fragments
        if next_atomFragment_attachment_loss_mask_size > 0:
            binary_next_attachment_point_losses = torch_scatter.scatter_add(out[3][select_multi_losses], mask_select_multi)
            next_atomFragment_attachment_loss = -torch.mean(torch.log(binary_next_attachment_point_losses + 1e-8))
            backprop_loss = backprop_loss + next_atomFragment_attachment_loss 
    
        else:
            next_atomFragment_attachment_loss = torch.tensor(float('NaN'))
            
        
        # predicting bond types of attachment points
        bond_loss = bond_cross_entropy(out[4], bond_type_mask)
        backprop_loss = backprop_loss + bond_loss 
                        
    else:
        next_atomFragment_loss = torch.tensor(float('NaN')) # Nan
        focal_attachment_loss = torch.tensor(float('NaN')) # Nan
        next_atomFragment_attachment_loss = torch.tensor(float('NaN'))  # Nan
        bond_loss = torch.tensor(float('NaN')) # Nan

        next_atomFragment_accuracy = torch.tensor(float('NaN')) # Nan
        
        focal_attachment_loss_mask_size = 0
        next_atomFragment_attachment_loss_mask_size = 0

        shape_loss = torch.tensor(float('NaN')) # Nan
    
    if training:
        backprop_loss.backward()
        optimizer.step()
    
    return batch_size, stop_loss.item(), next_atomFragment_loss.item(), focal_attachment_loss.item(), next_atomFragment_attachment_loss.item(), bond_loss.item(), sum(stop_mask.cpu().numpy()), focal_attachment_loss_mask_size, next_atomFragment_attachment_loss_mask_size, sum(stop_mask.cpu().numpy()), stop_accuracy.item(), next_atomFragment_accuracy.item(), batch.x.shape[0] // batch_size, batch.x_subgraph.shape[0] // batch_size, shape_loss.item(), KL_loss.item(), stop_shape_loss.item(), cosine_loss.item(), pairwise_property_loss.item(), N_pairwise_targets, mol_prop_loss.item()



logger('starting to train')
logger(f'train loader has approx. {len(train_loader)} batches')
val_logger(f'val loader has approx. {len(val_loader)} batches')

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma) if use_scheduler else None

stop_BCE = torch.nn.BCELoss()
next_atomFragment_cross_entropy = torch.nn.CrossEntropyLoss()
bond_cross_entropy = torch.nn.CrossEntropyLoss()

train_stop_loss = []
train_next_loss = []
train_focal_loss = []
train_attachment_loss = []
train_bond_loss = []
train_epoch_number = []
train_shape_loss = []
train_KL_loss = []
train_stop_shape_loss = []
train_cosine_loss = []
train_pairwise_property_loss = []
train_mol_prop_loss = []

val_stop_loss = []
val_next_loss = []
val_focal_loss = []
val_attachment_loss = []
val_bond_loss = []
val_epoch_number = []

val_stop_accuracy = []
val_next_accuracy = []
#val_mol_size = []
#val_subgraph_size = []
val_shape_loss = []
val_KL_loss = []
val_stop_shape_loss = []
val_cosine_loss = []
val_pairwise_property_loss = []
val_mol_prop_loss = []

interval = 5000
save_interval = 50000
scheduler_interval = 50000
validation_interval = 25000

stop_losses = []
next_losses = []
focal_losses = []
attachment_losses = []
bond_losses = []
batch_sizes = []
N_next_losses = []
N_focal_losses = []
N_attachment_losses = []
N_bond_losses = []
shape_losses = []
KL_losses = []
stop_shape_losses = []
cosine_losses = []
pairwise_property_losses = []
N_pairwise_targets_sizes = []
mol_prop_losses = []

logger(f"starting training with learning rate: {optimizer.param_groups[0]['lr']}")

for epoch in range(1, 1 + N_epochs):
    
    if validate_only == False:

        validate = False
    
        training = True
        model.train()
    
        for b, batch in enumerate(train_loader):
            batch_size, stopToken_loss, next_loss, focal_loss, attachment_loss, bond_loss, N_next_loss, N_focal_loss, N_attachment_loss, N_bond_loss, _, _, _, _, shape_loss, KL_loss, stop_shape_loss, cosine_loss, pairwise_property_loss, N_pairwise_targets, mol_prop_loss = loop(model, optimizer, batch, training = training, device = device, shape_penalty_factor = shape_penalty_factor, volume_distances = volume_distances, stop_shape_penalty = stop_shape_penalty, variational = variational, variational_mode = variational_mode, variational_GNN = variational_GNN, variational_GNN_mol = variational_GNN_mol, cosine_penalty = cosine_penalty, beta = beta, predict_pairwise_properties = predict_pairwise_properties, pairwise_property_factor = pairwise_property_factor, predict_mol_property = predict_mol_property, mol_property_factor = mol_property_factor)
            
            if iteration == 0:
                logger(f'stopToken_loss: {stopToken_loss}, next_loss: {next_loss}, focal_loss: {focal_loss}, attachment_loss: {attachment_loss}, bond_loss: {bond_loss}, shape_loss: {shape_loss}, KL_loss: {KL_loss}, stop_shape_loss: {stop_shape_loss}, pairwise_property_loss: {pairwise_property_loss}, mol_prop_loss: {mol_prop_loss}')

            if (iteration % 1000) == 0:
                gc.collect()
    
            batch_sizes.append(batch_size)
            stop_losses.append(stopToken_loss)
            next_losses.append(next_loss)
            focal_losses.append(focal_loss)
            attachment_losses.append(attachment_loss)
            bond_losses.append(bond_loss)
            N_next_losses.append(N_next_loss)
            N_focal_losses.append(N_focal_loss)
            N_attachment_losses.append(N_attachment_loss)
            N_bond_losses.append(N_bond_loss)
            shape_losses.append(shape_loss)
            KL_losses.append(KL_loss)
            stop_shape_losses.append(stop_shape_loss)
            cosine_losses.append(cosine_loss)
            pairwise_property_losses.append(pairwise_property_loss)
            N_pairwise_targets_sizes.append(N_pairwise_targets)
            mol_prop_losses.append(mol_prop_loss)
            
            if (iteration % interval) == 0:
    
                train_stop_loss.append(float(np.nansum(np.array(stop_losses) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
                train_next_loss.append(float(np.nansum(np.array(next_losses) * np.array(N_next_losses))) / sum(np.array(N_next_losses)))
                train_focal_loss.append(float(np.nansum(np.array(focal_losses) * np.array(N_focal_losses))) / sum(np.array(N_focal_losses)))
                train_attachment_loss.append(float(np.nansum(np.array(attachment_losses) * np.array(N_attachment_losses))) / sum(np.array(N_attachment_losses)))
                train_bond_loss.append(float(np.nansum(np.array(bond_losses) * np.array(N_bond_losses))) / sum(np.array(N_bond_losses)))
                train_epoch_number.append(epoch)
                train_shape_loss.append(float(np.nansum(np.array(shape_losses) * np.array(N_next_losses))) / sum(np.array(N_next_losses)))
                train_KL_loss.append(float(np.nansum(np.array(KL_losses) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
                train_stop_shape_loss.append(float(np.nansum(np.array(stop_shape_losses) * np.array(N_next_losses))) / sum(np.array(N_next_losses)))
                train_cosine_loss.append(float(np.nansum(np.array(cosine_losses) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
                train_pairwise_property_loss.append(float(np.nansum(np.array(pairwise_property_losses) * np.array(N_pairwise_targets_sizes))) / sum(np.array(N_pairwise_targets_sizes)))
                train_mol_prop_loss.append(float(np.nansum(np.array(mol_prop_losses) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
                

                logger(f'iteration: {iteration}, epoch: {epoch}, batch: {b}, stop_loss: {train_stop_loss[-1]}, next_loss: {train_next_loss[-1]}, focal_loss: {train_focal_loss[-1]}, attachment_loss: {train_attachment_loss[-1]}, bond_loss: {train_bond_loss[-1]}, shape_loss: {train_shape_loss[-1]}, KL loss: {train_KL_loss[-1]}, stop_shape_loss: {train_stop_shape_loss[-1]}, cosine_loss: {train_cosine_loss[-1]}, pairwise_property_loss: {train_pairwise_property_loss[-1]}, mol_prop_loss: {train_mol_prop_loss[-1]}' )  
                
                batch_sizes = []
                stop_losses = []
                next_losses = []
                focal_losses = []
                attachment_losses = []
                bond_losses = []
                N_next_losses = []
                N_focal_losses = []
                N_attachment_losses = []
                N_bond_losses = []
                shape_losses = []
                KL_losses = []
                stop_shape_losses = []
                cosine_losses = []
                pairwise_property_losses = []
                N_pairwise_targets_sizes = []
                mol_prop_losses = []
    
            if (save) & ((iteration % save_interval) == 0):
                logger(f'saving model {int(iteration / save_interval)}...')
                torch.save(model.state_dict(), PATH + f'saved_models/model_{int(iteration / save_interval)}.pt')
    
            if (use_scheduler == True) & (iteration % scheduler_interval == 0):
                scheduler.step()
                logger(f"learning rate reduced to: {optimizer.param_groups[0]['lr']}")
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    use_scheduler = False
                    
            if ((variational == True) | (variational_GNN == True) | (variational_GNN_mol == True)) & (iteration % beta_interval == 0):
                beta_iteration += 1
                beta = float(beta_schedule[beta_iteration])
                logger(f"beta increased to: {beta}. New beta iteration: {beta_iteration}")
    
            iteration += 1
    
            if (iteration - 1) % validation_interval == 0:
                validate = True # validate model after epoch chunk finishes
    
    else:
        validate = True

    if validate == False:
        continue

    logger(f'validating model at iteration: {iteration}')

    val_stop_losses = []
    val_next_losses = []
    val_focal_losses = []
    val_attachment_losses = []
    val_bond_losses = []
    val_batch_sizes = []
    val_N_next_losses = []
    val_N_focal_losses = []
    val_N_attachment_losses = []
    val_N_bond_losses = []
    
    val_stop_accuracies = []
    val_next_accuracies = []
    #val_mol_sizes = []
    #val_subgraph_sizes = []
    val_shape_losses = []
    val_KL_losses = []
    val_stop_shape_losses = []
    val_cosine_losses = []
    val_pairwise_property_losses = []
    val_N_pairwise_targets_sizes = []
    val_mol_prop_losses = []

    training = False
    model.eval()

    for b, batch in enumerate(val_loader):
        batch_size, stopToken_loss, next_loss, focal_loss, attachment_loss, bond_loss, N_next_loss, N_focal_loss, N_attachment_loss, N_bond_loss, stop_accuracy, next_atomFragment_accuracy, mol_size, subgraph_size, shape_loss, KL_loss, stop_shape_loss, cosine_loss, pairwise_property_loss, N_pairwise_targets, mol_prop_loss = loop(model, optimizer, batch, training = training, device = device, shape_penalty_factor = shape_penalty_factor, volume_distances = volume_distances, stop_shape_penalty = stop_shape_penalty, variational = variational, variational_mode = variational_mode, variational_GNN = variational_GNN, variational_GNN_mol = variational_GNN_mol, cosine_penalty = cosine_penalty, beta = beta, predict_pairwise_properties = predict_pairwise_properties, pairwise_property_factor = pairwise_property_factor, predict_mol_property = predict_mol_property, mol_property_factor = mol_property_factor)
        
        val_batch_sizes.append(batch_size)
        val_stop_losses.append(stopToken_loss)
        val_next_losses.append(next_loss)
        val_focal_losses.append(focal_loss)
        val_attachment_losses.append(attachment_loss)
        val_bond_losses.append(bond_loss)
        val_N_next_losses.append(N_next_loss)
        val_N_focal_losses.append(N_focal_loss)
        val_N_attachment_losses.append(N_attachment_loss)
        val_N_bond_losses.append(N_bond_loss)
        
        val_stop_accuracies.append(stop_accuracy)
        val_next_accuracies.append(next_atomFragment_accuracy)
        #val_mol_sizes.append(mol_size)
        #val_subgraph_sizes.append(subgraph_size)
        val_shape_losses.append(shape_loss)
        val_KL_losses.append(KL_loss)
        val_stop_shape_losses.append(stop_shape_loss)
        val_cosine_losses.append(cosine_loss)
        val_pairwise_property_losses.append(pairwise_property_loss)
        val_N_pairwise_targets_sizes.append(N_pairwise_targets)
        val_mol_prop_losses.append(mol_prop_loss)
    
    val_stop_loss.append( float(np.nansum(np.array(val_stop_losses) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_next_loss.append(float(np.nansum(np.array(val_next_losses) * np.array(val_N_next_losses))) / sum(np.array(val_N_next_losses)))
    val_focal_loss.append(float(np.nansum(np.array(val_focal_losses) * np.array(val_N_focal_losses))) / sum(np.array(val_N_focal_losses)))
    val_attachment_loss.append(float(np.nansum(np.array(val_attachment_losses) * np.array(val_N_attachment_losses))) / sum(np.array(val_N_attachment_losses)))
    val_bond_loss.append(float(np.nansum(np.array(val_bond_losses) * np.array(val_N_bond_losses))) / sum(np.array(val_N_bond_losses)))
    val_epoch_number.append(epoch)

    val_stop_accuracy.append(float(np.nansum(np.array(val_stop_accuracies) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_next_accuracy.append(float(np.nansum(np.array(val_next_accuracies) * np.array(val_N_next_losses))) / sum(np.array(val_N_next_losses)))
    #val_mol_size.append()
    #val_subgraph_size.append()
    val_shape_loss.append(float(np.nansum(np.array(val_shape_losses) * np.array(val_N_next_losses))) / sum(np.array(val_N_next_losses)))
    val_KL_loss.append( float(np.nansum(np.array(val_KL_losses) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_stop_shape_loss.append(float(np.nansum(np.array(val_stop_shape_losses) * np.array(val_N_next_losses))) / sum(np.array(val_N_next_losses)))
    val_cosine_loss.append( float(np.nansum(np.array(val_cosine_losses) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_pairwise_property_loss.append(float(np.nansum(np.array(val_pairwise_property_losses) * np.array(val_N_pairwise_targets_sizes))) / sum(np.array(val_N_pairwise_targets_sizes)))
    val_mol_prop_loss.append( float(np.nansum(np.array(val_mol_prop_losses) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    
    
    val_logger(f'iteration: {iteration}, stop_loss:{val_stop_loss[-1]}, next_loss: {val_next_loss[-1]}, focal_loss: {val_focal_loss[-1]}, attachment_loss: {val_attachment_loss[-1]}, bond_loss: {val_bond_loss[-1]}, stop_accuracy: {val_stop_accuracy[-1]}, next_accuracy: {val_next_accuracy[-1]}, shape_loss: {val_shape_loss[-1]}, KL_loss: {val_KL_loss[-1]}, stop_shape_loss: {val_stop_shape_loss[-1]}, cosine_loss: {val_cosine_loss[-1]}, pairwise_property_loss: {val_pairwise_property_loss[-1]}, mol_prop_loss: {val_mol_prop_loss[-1]}')
    
    logger(f'finished validating...\n')

