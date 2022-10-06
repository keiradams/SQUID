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
from rdkit.Geometry import Point3D

import os
import shutil

import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import *
from utils.scorer_datasets_and_loaders import *

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

mix_node_inv_to_equi = True
mix_shape_to_nodes = True
ablate_HvarCat = False

variational = False # for Z_inv or Z_equi
variational_mode = 'inv' # both, equi, or inv
variational_GNN = False # for GNN-encoded atom embeddings

beta_schedule = np.concatenate((np.logspace(-5, -1, 100), np.ones(400)*1e-1))
beta_interval = 10000 # update beta every 10000 iterations (batches)

name = 'training_scorer'

PATH = 'results_' + name + '/'
output_file = PATH + name

# change these to load a checkpoint model
model_state = ''
learning_rate_state = None
iteration = 1
beta_iteration = 0

beta = float(beta_schedule[beta_iteration]) if ((variational == True) | (variational_GNN == True)) else None

# Data Augmentation
dihedral_var = 5.0 # 5.0

# HYPERPARAMETERS
input_nf = 45
edges_in_d = 5
n_knn = 10
conv_dims = [32, 32, 64, 128]
num_components = 64
fragment_library_dim = 64
N_fragment_layers = 3
N_members = 125 - 1
EGNN_layer_dim = 64
N_EGNN_layers = 3
output_MLP_hidden_dim = 64

append_noise = False
learned_noise = False

pooling_MLP = False
shared_encoders = False
subtract_latent_space = True

target_batch_size = 32
N_rot = 10

N_points = 5


if learning_rate_state is not None:
    lr = learning_rate_state
else:
    lr = 0.0005

min_lr = 0.0005 / 50.
use_scheduler = True
gamma = 0.9
num_workers = 16
N_epochs = 10 * 40

chunks = 20

seed = 0
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

if save:
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        os.makedirs(PATH + 'saved_models/')
    
    shutil.copyfile('train_scorer.py', PATH + 'train_scorer.py')

def logger(text, file = output_file + '_training_log.txt'):
    if save:
        with open(file, 'a') as f:
            f.write(text + '\n')
    else:
        print(text + '\n')
        
def val_logger(text, file = output_file + '_validation_log.txt'):
    if save:
        with open(file, 'a') as f:
            f.write(text + '\n')
    else:
        print(text + '\n')

logger('reading databases')
AtomFragment_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl')
AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop = True)

filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')

# replace mols with artificial mols
MOSES2_training_val_filtered_database_artificial_mols = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database_artificial_mols.pkl')
filtered_database['rdkit_mol_cistrans_stereo'] = MOSES2_training_val_filtered_database_artificial_mols.artificial_mols

# I need to remove duplicates in order for the indexing to work
unmerged_future_rocs_db = pd.read_pickle('data/MOSES2/MOSES2_training_val_canonical_terminalSeeds_unmerged_future_rocs_database_all_reduced.pkl').drop_duplicates(['original_index', 'dihedral_indices', 'positions_before_sorted']).reset_index(drop = True)
unmerged_future_rocs_db['max_future_rocs_index'] = range(0, len(unmerged_future_rocs_db))

train_smiles_df = pd.read_csv('data/MOSES2/MOSES2_train_smiles_split.csv')
train_smiles = set(train_smiles_df.SMILES_nostereo)
train_db_mol = filtered_database.loc[filtered_database['SMILES_nostereo'].isin(train_smiles)].reset_index(drop = True)
train_database = train_db_mol[['original_index', 'N_atoms', 'has_conf', 'rdkit_mol_cistrans_stereo']].merge(unmerged_future_rocs_db, on='original_index')

val_smiles_df = pd.read_csv('data/MOSES2/MOSES2_val_smiles_split.csv')
val_smiles = set(val_smiles_df.SMILES_nostereo)
val_db_mol = filtered_database.loc[filtered_database['SMILES_nostereo'].isin(val_smiles)].reset_index(drop = True)
val_database = val_db_mol[['original_index', 'N_atoms', 'has_conf', 'rdkit_mol_cistrans_stereo']].merge(unmerged_future_rocs_db, on='original_index')


edge_index_array = np.load('data/MOSES2/MOSES2_training_val_edge_index_array.npy')
edge_features_array = np.load('data/MOSES2/MOSES2_training_val_edge_features_array.npy')
node_features_array = np.load('data/MOSES2/MOSES2_training_val_node_features_array.npy')

xyz_array = np.load('data/MOSES2/MOSES2_training_val_xyz_artificial_array.npy')

atom_fragment_associations_array = np.load('data/MOSES2/MOSES2_training_val_atom_fragment_associations_array.npy')
atom_fragment_associations_array = atom_fragment_associations_array - 1

atoms_pointer = np.load('data/MOSES2/MOSES2_training_val_atoms_pointer.npy')
bonds_pointer = np.load('data/MOSES2/MOSES2_training_val_bonds_pointer.npy')

max_future_rocs = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_max_future_rocs.npy')
max_future_rocs_evaluated_dihedrals = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals.npy')


def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

mols = list(train_database.rdkit_mol_cistrans_stereo)
original_index = np.array(train_database.original_index)
train_max_future_rocs_index = np.array(train_database.max_future_rocs_index)

dihedral_indices_array = np.array(flatten(train_database.dihedral_indices))
dihedral_indices_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.dihedral_indices), total = len(train_database)):
    dihedral_indices_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
dihedral_indices_pointer[-1] = s

indices_partial_before_array = np.array(flatten(train_database.positions_before_sorted))
indices_partial_before_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.positions_before_sorted), total = len(train_database)):
    indices_partial_before_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
indices_partial_before_pointer[-1] = s

indices_partial_after_array = np.array(flatten(train_database.positions_after_sorted))
indices_partial_after_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.positions_after_sorted), total = len(train_database)):
    indices_partial_after_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
indices_partial_after_pointer[-1] = s

query_indices_array = np.array(flatten(train_database.query_indices))
query_indices_pointer = np.zeros(len(train_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(train_database.query_indices), total = len(train_database)):
    query_indices_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
query_indices_pointer[-1] = s


val_mols = list(val_database.rdkit_mol_cistrans_stereo)
val_original_index = np.array(val_database.original_index)
val_max_future_rocs_index = np.array(val_database.max_future_rocs_index)

val_dihedral_indices_array = np.array(flatten(val_database.dihedral_indices))
val_dihedral_indices_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.dihedral_indices), total = len(val_database)):
    val_dihedral_indices_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
val_dihedral_indices_pointer[-1] = s

val_indices_partial_before_array = np.array(flatten(val_database.positions_before_sorted))
val_indices_partial_before_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.positions_before_sorted), total = len(val_database)):
    val_indices_partial_before_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
val_indices_partial_before_pointer[-1] = s

val_indices_partial_after_array = np.array(flatten(val_database.positions_after_sorted))
val_indices_partial_after_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.positions_after_sorted), total = len(val_database)):
    val_indices_partial_after_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
val_indices_partial_after_pointer[-1] = s

val_query_indices_array = np.array(flatten(val_database.query_indices))
val_query_indices_pointer = np.zeros(len(val_database) + 1, dtype = int)
s = 0
for i, p in tqdm(enumerate(val_database.query_indices), total = len(val_database)):
    val_query_indices_pointer[i] = s
    if p == -1:
        s += 1
    else:
        s += len(p)
val_query_indices_pointer[-1] = s


logger('initializing model')
model = ROCS_Model_Point_Cloud(
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
    
    mix_node_inv_to_equi = mix_node_inv_to_equi,
    mix_shape_to_nodes = mix_shape_to_nodes,
    ablate_HvarCat = ablate_HvarCat,
    
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


logger('creating dataloaders')

library_dataset = AtomFragmentLibrary(AtomFragment_database)
library_loader = torch_geometric.data.DataLoader(
    library_dataset, 
    shuffle = False, 
    batch_size = len(library_dataset), 
    num_workers = 0,
)
fragment_batch = next(iter(library_loader))
fragment_batch = fragment_batch.to(device)


train_sampler = VNNBatchSampler(train_database, target_batch_size, chunks = chunks)
train_dataset = ROCSDataset_point_cloud(
    mols = mols,
    max_future_rocs = max_future_rocs,
    max_future_rocs_evaluated_dihedrals = max_future_rocs_evaluated_dihedrals,
    max_future_rocs_index = train_max_future_rocs_index,

    original_index = original_index, 

    edge_index_array = edge_index_array, 
    edge_features_array = edge_features_array, 
    node_features_array = node_features_array, 
    xyz_array = xyz_array,
    atom_fragment_associations_array = atom_fragment_associations_array,
    atoms_pointer = atoms_pointer, 
    bonds_pointer = bonds_pointer, 

    dihedral_indices_array = dihedral_indices_array,
    dihedral_indices_pointer = dihedral_indices_pointer,
    indices_partial_before_array = indices_partial_before_array,
    indices_partial_before_pointer = indices_partial_before_pointer, 
    indices_partial_after_array = indices_partial_after_array,
    indices_partial_after_pointer  = indices_partial_after_pointer,
    query_indices_array = query_indices_array, 
    query_indices_pointer = query_indices_pointer, 
    
    N_points = N_points, 
    N_rot = N_rot,

    dihedral_var = dihedral_var,
)
train_loader = torch_geometric.data.DataLoader(
    train_dataset, 
    batch_sampler = train_sampler, 
    num_workers = num_workers, 
    follow_batch = ['x', 'x_subgraph'])


val_sampler = VNNBatchSampler(val_database, target_batch_size, chunks = 25)
val_dataset = ROCSDataset_point_cloud(
    mols = val_mols, 
    
    max_future_rocs = max_future_rocs,
    max_future_rocs_evaluated_dihedrals = max_future_rocs_evaluated_dihedrals,
    max_future_rocs_index = val_max_future_rocs_index,
    
    original_index = val_original_index, 

    edge_index_array = edge_index_array, 
    edge_features_array = edge_features_array, 
    node_features_array = node_features_array, 
    xyz_array = xyz_array, 
    atom_fragment_associations_array = atom_fragment_associations_array,
    atoms_pointer = atoms_pointer, 
    bonds_pointer = bonds_pointer, 

    dihedral_indices_array = val_dihedral_indices_array,
    dihedral_indices_pointer = val_dihedral_indices_pointer,
    indices_partial_before_array = val_indices_partial_before_array,
    indices_partial_before_pointer = val_indices_partial_before_pointer, 
    indices_partial_after_array = val_indices_partial_after_array,
    indices_partial_after_pointer  = val_indices_partial_after_pointer,
    query_indices_array = val_query_indices_array, 
    query_indices_pointer = val_query_indices_pointer, 
    
    N_points = N_points, 
    N_rot = N_rot,

    dihedral_var = dihedral_var,
)
val_loader = torch_geometric.data.DataLoader(
    val_dataset, 
    batch_sampler = val_sampler, 
    num_workers = num_workers, 
    follow_batch = ['x', 'x_subgraph'])


def loop(model, optimizer, batch, training = True, device = torch.device('cpu'), N_rot = 10, variational_GNN = False, beta = 0.0):
    data, rocs = batch
    
    batch_size = data.subgraph_size.shape[0]
    
    if training:
        optimizer.zero_grad()
    
    rocs = rocs.reshape(-1) 
    
    query_indices_rel_to_partial = data.query_index_subgraph
    query_indices_batch = data.new_batch_subgraph[query_indices_rel_to_partial]
    
    data = data.to(device)
    rocs = rocs.float().to(device)
    
    args = (
        batch_size, 
        
        data.x.float(), 
        data.edge_index, 
        data.edge_attr.float(), 
        data.pos.float(), 
        data.cloud.float(), 
        data.cloud_indices, 
        data.atom_fragment_associations, 
        
        data.x_subgraph.float(), 
        data.edge_index_subgraph, 
        data.edge_attr_subgraph.float(),
        data.pos_subgraph.float(), 
        data.cloud_subgraph.float(), 
        data.cloud_indices_subgraph, 
        data.atom_fragment_associations_subgraph, 
        
        query_indices_rel_to_partial.to(device), 
        query_indices_batch.to(device), 
        fragment_batch,
    )
        
    if not training:
        with torch.no_grad():
            out_ = model(*args, device = device)
    else:
        out_ = model(*args, device = device)
    
    out = out_[0] # predicted scores
    out = torch.sigmoid(out)
    MSE_loss = torch.mean(torch.square(out.squeeze() - rocs.squeeze()))
    backprop_loss = MSE_loss
    
    if variational_GNN: # since batches contain molecules with same # of atoms, we don't need to do any additional averaging
        h_mean, h_std = out_[5], out_[6]
        KL_unreduced = 0.5 * (torch.sum(h_mean**2.0, dim = 1) + torch.sum(h_std**2.0, dim = 1) - torch.sum(torch.log(h_std**2.0) + 1.0, dim = 1))
        KL_loss = torch.mean(KL_unreduced)
        backprop_loss = backprop_loss + beta * KL_loss
        
    else: # no variational components in encoder
        KL_loss = torch.tensor(float('NaN')) # Nan
        
        
    mae = torch.mean(torch.abs(out.squeeze() - rocs.squeeze()))
    acc = torch.mean((torch.argmax(out.squeeze().reshape(-1, N_rot), dim = 1) == torch.argmax(rocs.squeeze().reshape(-1, N_rot), dim = 1)).type(torch.float))

    if training:
        backprop_loss.backward()
        optimizer.step()
    
    return batch_size, MSE_loss.item(), mae.item(), acc.item(), KL_loss.item()


logger('starting to train')
len_train_loader = len(train_loader)
len_val_loader = len(val_loader)
logger(f'train loader has approx. {len_train_loader} batches')
val_logger(f'val loader has approx. {len_val_loader} batches')

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma) if use_scheduler else None

train_loss = []
train_KL_loss = []
train_mae = []
train_acc = []
train_epoch_number = []

val_loss = []
val_KL_loss = []
val_mae = []
val_acc = []
val_epoch_number = []


interval = 5000
save_interval = 30000
scheduler_interval = 50000
validation_interval = 25000

losses = []
KL_losses = []
MAEs = []
accs = []
batch_sizes = []

logger(f"starting training with learning rate: {optimizer.param_groups[0]['lr']}")

for epoch in range(1, 1 + N_epochs):
    
    validate = False

    training = True
    model.train()

    for b, batch in enumerate(train_loader):
            
        batch_size, loss, mae, acc, KL_loss = loop(model, optimizer, batch, training = training, device = device, N_rot = N_rot, variational_GNN = variational_GNN, beta = beta)

        if (iteration % 1000) == 0:
            gc.collect()
        
        batch_sizes.append(batch_size)
        MAEs.append(mae)
        accs.append(acc)
        losses.append(loss)
        KL_losses.append(KL_loss)
        
        if iteration % interval == 0:

            train_loss.append(float(np.nansum(np.array(losses) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
            train_KL_loss.append(float(np.nansum(np.array(KL_losses) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))

            train_mae.append(float(np.nansum(np.array(MAEs) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
            train_acc.append(float(np.nansum(np.array(accs) * np.array(batch_sizes))) / sum(np.array(batch_sizes)))
            train_epoch_number.append(epoch)
            
            logger(f'iteration: {iteration}, epoch: {epoch}, batch: {b}, loss: {train_loss[-1]}, MAE: {train_mae[-1]}, acc: {train_acc[-1]}, KL_loss: {train_KL_loss[-1]}' )  
            losses = []
            KL_losses = []
            MAEs = []
            accs = []
            batch_sizes = []

        if (save) & ((iteration % save_interval) == 0):
            logger(f'saving model {int(iteration / save_interval)}...')
            torch.save(model.state_dict(), PATH + f'saved_models/rocs_model_{int(iteration / save_interval)}.pt')

        if (use_scheduler == True) & (iteration % scheduler_interval == 0):
            scheduler.step()
            logger(f"learning rate reduced to: {optimizer.param_groups[0]['lr']}")
            if optimizer.param_groups[0]['lr'] <= min_lr:
                use_scheduler = False
        
        if ((variational == True) | (variational_GNN == True)) & (iteration % beta_interval == 0):
            beta_iteration += 1
            beta = float(beta_schedule[beta_iteration])
            logger(f"beta increased to: {beta}")

        iteration += 1

        if (iteration - 1) % validation_interval == 0:
            validate = True # validate model after epoch chunk finishes
    

    if validate == False:
        continue

    logger(f'validating model at iteration: {iteration - 1}')
    
    training = False
    val_losses = []
    val_KL_losses = []
    val_MAEs = []
    val_accs = []
    val_batch_sizes = []
    model.eval()
    for b, batch in enumerate(val_loader): # with chunks = 25, we only evaluate on a random 4% of the entire validation set
        batch_size, loss, mae, acc, KL_loss = loop(model, None, batch, training = training, device = device, N_rot = N_rot, variational_GNN = variational_GNN, beta = beta)

        if (b % 1000) == 0:
            gc.collect()
        
        val_batch_sizes.append(batch_size)
        val_MAEs.append(mae)
        val_accs.append(acc)
        val_losses.append(loss)
        val_KL_losses.append(KL_loss)
           
    val_loss.append(float(np.nansum(np.array(val_losses) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_KL_loss.append(float(np.nansum(np.array(val_KL_losses) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))

    val_mae.append(float(np.nansum(np.array(val_MAEs) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_acc.append(float(np.nansum(np.array(val_accs) * np.array(val_batch_sizes))) / sum(np.array(val_batch_sizes)))
    val_epoch_number.append(epoch)

    val_logger(f'iteration: {iteration - 1}, loss: {val_loss[-1]}, MAE: {val_mae[-1]}, acc: {val_acc[-1]}, KL_loss: {val_KL_loss[-1]}')
    
