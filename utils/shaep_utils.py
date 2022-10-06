import os
import random

import rdkit
import rdkit.Chem
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit import Geometry

from utils.general_utils import *

# should redesign to make exactly analogous to ROCS_shape_overlap
def shape_align(reference, query, shaep_path = '../../software', remove_files = True, ID = ''):
    
    if not os.path.exists('shaep_objects_temp'):
        os.makedirs('shaep_objects_temp')
    
    job_number = random.randint(0, 10000000)
    
    rdkit.Chem.rdmolfiles.MolToV3KMolFile(reference, f'shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol')
    rdkit.Chem.rdmolfiles.MolToV3KMolFile(query, f'shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol')
    
    os.system(f"{shaep_path}/shaep --onlyshape -q shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol -s shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt >/dev/null 2>&1")

    suppl = rdkit.Chem.rdmolfiles.ForwardSDMolSupplier(f'shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf')
    mol = next(suppl)
    
    if remove_files:
        os.system(f'rm shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}_hits.txt')
    
    rocs = get_ROCS(torch.as_tensor(mol.GetConformer().GetPositions()), torch.as_tensor(reference.GetConformer().GetPositions()))
    
    return mol, float(mol.GetProp('Similarity_shape')), rocs.item()


def ESP_shape_align(reference, query, shaep_path = '../../software', remove_files = True, ID = ''):
    
    if not os.path.exists('shaep_objects_temp'):
        os.makedirs('shaep_objects_temp')
        
    job_number = random.randint(0, 10000000)
    
    rdkit.Chem.rdmolfiles.MolToV3KMolFile(reference, f'shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol')
    rdkit.Chem.rdmolfiles.MolToV3KMolFile(query, f'shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol')
    
    os.system(f"{shaep_path}/shaep -q shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol -s shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt >/dev/null 2>&1")
    
    suppl = rdkit.Chem.rdmolfiles.ForwardSDMolSupplier(f'shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf')
    mol = next(suppl)
    
    if remove_files:
        os.system(f'rm shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}_hits.txt')
    
    rocs = get_ROCS(torch.as_tensor(mol.GetConformer().GetPositions()), torch.as_tensor(reference.GetConformer().GetPositions()))
    
    return mol, float(mol.GetProp('Similarity_shape')), float(mol.GetProp('Similarity_ESP')), rocs.item()
