from multiprocessing import Pool

import pickle
import os
import pandas as pd
import numpy as np
import itertools
import subprocess
from tqdm import tqdm
from copy import deepcopy
import random
import math
import torch
import pickle

from functools import partial
from itertools import repeat

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
from rdkit.Chem import rdMolTransforms
import rdkit.Chem.rdMolAlign
from rdkit.Geometry import Point3D
from rdkit import Geometry
import rdkit.Chem.rdShapeHelpers
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolAlign

from tdc import Oracle

#'GSK3B'
#'JNK3'
#'Osimertinib_MPO'
#'Sitagliptin_MPO'
#'Celecoxib_Rediscovery'
#'Thiothixene_Rediscovery'

oracle = Oracle(name = 'GSK3B') # manually change

def score_oracle(mol_list):
    if type(mol_list[0]) is str:
        smiles_list = mol_list
    else:
        smiles_list = [rdkit.Chem.MolToSmiles(m) for m in mol_list]
    scores = oracle(smiles_list)
    return scores


def parallel_scoring(mol_list):
    
    scores = score_oracle(mol_list)
    
    return scores


if __name__ == '__main__':
    
    oracle_name = 'GSK3B'
    
    print('reading data...')
    
    test_mol_df = pd.read_pickle('data/MOSES2/test_MOSES_filtered_artificial_mols.pkl')
    query_mols = list(test_mol_df.artificial_mol)
    
    print(f'screening {len(query_mols)} total molecules')
    
    num_per_chunk = 2000
    
    N_chunks = int(math.ceil(len(query_mols) / num_per_chunk))
    
    query_chunks = [query_mols[num_per_chunk*i:num_per_chunk*(i+1)] for i in range(N_chunks)]
    
    print(f'there are {N_chunks} total chunks')
    
    print('processing chunks...')
    chunk_scores = []
    pool = Pool(16)    
    for scores in tqdm(pool.imap(partial(parallel_scoring), query_chunks), total = N_chunks):
        chunk_scores.append(scores)
    pool.close()
    pool.join()
    
    flattened_scores = [item for sublist in chunk_scores for item in sublist]
    
    score_array = np.array(flattened_scores)
    
    np.save(f'optimization_results_MOSES2/test_set_scores_{oracle_name}.npy', score_array)
    

