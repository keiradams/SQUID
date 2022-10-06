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

from multiprocessing import Pool

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

import openeye
import os
from openeye.oechem import *
from openeye.oeiupac import *
from openeye.oeomega import *
from openeye.oeshape import *
from openeye import oeshape
from openeye.oedepict import *
license_filename = './oe_license.txt'

if os.path.isfile(license_filename):
    license_file = open(license_filename, 'r')
    openeye.OEAddLicenseData(license_file.read())
    license_file.close()
    assert openeye.oechem.OEChemIsLicensed()
else:
    raise Exception("Error: Your OpenEye license is not readable; please check your filename and that you have mounted your Google Drive")


def VAB_2nd_order(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    R2 = (torch.cdist(centers_1, centers_2)**2.0).T
    prefactor1_prod_prefactor2 = prefactors_1 * prefactors_2.unsqueeze(1)
    alpha1_prod_alpha2 = alphas_1 * alphas_2.unsqueeze(1)
    alpha1_sum_alpha2 = alphas_1 + alphas_2.unsqueeze(1)

    VAB_2nd_order = torch.sum(np.pi**(1.5) * prefactor1_prod_prefactor2 * torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2) / (alpha1_sum_alpha2**(1.5)))
    return VAB_2nd_order

def shape_tanimoto(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    VAA = VAB_2nd_order(centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1)
    VBB = VAB_2nd_order(centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2)
    VAB = VAB_2nd_order(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return VAB / (VAA + VBB - VAB)

def get_ROCS(centers_1, centers_2, prefactor = 0.8, alpha = 0.81):
    #centers_1 = torch.tensor(centers_1)
    #centers_2 = torch.tensor(centers_2)
    prefactors_1 = torch.ones(centers_1.shape[0]) * prefactor 
    prefactors_2 = torch.ones(centers_2.shape[0]) * prefactor
    alphas_1 = torch.ones(prefactors_1.shape[0]) * alpha
    alphas_2 = torch.ones(prefactors_2.shape[0]) * alpha

    tanimoto = shape_tanimoto(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return tanimoto

def get_ROCS_mols(mol_1, mol_2, prefactor = 0.8, alpha = 0.81):
    pos_1 = torch.from_numpy(mol_1.GetConformer().GetPositions())
    pos_2 = torch.from_numpy(mol_2.GetConformer().GetPositions())
    return get_ROCS(pos_1, pos_2, prefactor = prefactor, alpha = alpha)



from rdkit import Chem
def oemol_from_rdmol(rdkitmol):
    rdmol = deepcopy(rdkitmol)
    """
    Creates an openeye molecule object that is identical to the input rdkit molecule
    """
    # RDK automatically includes explicit hydrogens in its SMILES patterns
    #print("Starting molecule: ", Chem.MolToSmiles(Chem.RemoveHs(rdmol))) 
    
    # openeye stores bond orders as integers regardless of aromaticity
    # in order to properly extract these, we need to have the "Kekulized" version of the rdkit mol
    kekul_mol = Chem.Mol(rdmol)
    Chem.Kekulize(kekul_mol, True)
    
    oemol = oechem.OEMol()
    map_atoms = dict() # {rd_idx: oe_atom}
    
    # setting chirality in openey requires using neighbor atoms
    # therefore we can't do it until after the atoms and bonds are all added
    chiral_atoms = dict() # {rd_idx: openeye chirality}
    for rda in rdmol.GetAtoms():
        rd_idx = rda.GetIdx()
        
        # create a new atom
        oe_a = oemol.NewAtom(rda.GetAtomicNum())
        map_atoms[rd_idx] = oe_a
        oe_a.SetFormalCharge(rda.GetFormalCharge())
        oe_a.SetAromatic(rda.GetIsAromatic())

        # If chiral, store the chirality to be set later
        tag = rda.GetChiralTag() 
        if tag == Chem.CHI_TETRAHEDRAL_CCW:
            chiral_atoms[rd_idx] = oechem.OECIPAtomStereo_R
        if tag == Chem.CHI_TETRAHEDRAL_CW:
            chiral_atoms[rd_idx] = oechem.OECIPAtomStereo_S

    # Similar to chirality, stereochemistry of bonds in OE is set relative to their neighbors
    stereo_bonds = list()
    # stereo_bonds stores tuples in the form (oe_bond, rd_idx1, rd_idx2, OE stereo specification)
    # where rd_idx1 and 2 are the atoms on the outside of the bond
    # i.e. Cl and F in the example above
    aro_bond = 0
    for rdb in rdmol.GetBonds():
        a1 = rdb.GetBeginAtomIdx()
        a2 = rdb.GetEndAtomIdx()
        
        # create a new bond
        newbond = oemol.NewBond(map_atoms[a1], map_atoms[a2])
        
        order = rdb.GetBondTypeAsDouble()
        if order == 1.5: 
            # get the bond order for this bond in the kekulized molecule
            order = kekul_mol.GetBondWithIdx(rdb.GetIdx()).GetBondTypeAsDouble()
            newbond.SetAromatic(True)
        else:
            newbond.SetAromatic(False)
        newbond.SetOrder(int(order))

        # determine if stereochemistry is needed
        tag = rdb.GetStereo()
        if tag == Chem.BondStereo.STEREOCIS or tag == Chem.BondStereo.STEREOZ:
            stereo_atoms = rdb.GetStereoAtoms()
            stereo_bonds.append((newbond, stereo_atoms[0], stereo_atoms[1], oechem.OEBondStereo_Cis))
            
            bond2 = rdmol.GetBondBetweenAtoms(stereo_atoms[0], a1)
            bond4 = rdmol.GetBondBetweenAtoms(stereo_atoms[1], a2)
            print(tag, bond2.GetBondDir(), bond4.GetBondDir())
        if tag == Chem.BondStereo.STEREOTRANS or tag == Chem.BondStereo.STEREOE:
            stereo_atoms = rdb.GetStereoAtoms()
            stereo_bonds.append((newbond, stereo_atoms[0], stereo_atoms[1], oechem.OEBondStereo_Trans))
            bond2 = rdmol.GetBondBetweenAtoms(stereo_atoms[0], a1)
            bond4 = rdmol.GetBondBetweenAtoms(stereo_atoms[1], a2)
            print(tag, bond2.GetBondDir(), bond4.GetBondDir())
            
    # Now that all of the atoms are connected we can set stereochemistry
    # starting with atom chirality
    for rd_idx, chirality in chiral_atoms.items():
        # chirality is set relative to neighbors, so we will get neighboring atoms
        # assign Right handed direction, check the cip stereochemistry
        # if the cip stereochemistry isn't correct then we'll set left and double check
        
        oea = map_atoms[rd_idx]
        neighs = [n for n in oea.GetAtoms()]
        # incase you look at the documentation oe has two options for handedness for example:
        # oechem.OEAtomStereo_Left == oechem.OEAtomStereo_LeftHanded
        oea.SetStereo(neighs, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Right)
        cip = oechem.OEPerceiveCIPStereo(oemol, oea)
        if cip != chirality:
            oea.SetStereo(neighs, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Left)
            new_cip = oechem.OEPerceiveCIPStereo(oemol, oea)
            if new_cip != chirality:
                # Note, I haven't seen this happen yet, but it shouldn't be a problem since there 
                # is only 2 directions for handedness and we're only running this for chiral atoms
                #print("PANIC!")
                pass

    # Set stereochemistry using the reference atoms extracted above
    for oeb, idx1, idx2, oestereo in stereo_bonds:
        oeb.SetStereo([map_atoms[idx1], map_atoms[idx2]], oechem.OEBondStereo_CisTrans, oestereo)

    # If the rdmol has a conformer, add its coordinates to the oemol
    # Note, this currently only adds the first conformer, it will need to be adjusted if the
    # you wanted to convert multiple sets of coordinates
    if rdmol.GetConformers():
        #print("found an rdmol conformer")
        conf = rdmol.GetConformer()
        for rd_idx, oeatom in map_atoms.items():
            coords = conf.GetAtomPosition(rd_idx)
            oemol.SetCoords(oeatom, oechem.OEFloatArray(coords))
        
    # If RDMol has a title save it
    if rdmol.HasProp("_Name"):
        oemol.SetTitle(rdmol.GetProp("_Name"))
        
    # Clean Up phase
    # The only feature of a molecule that wasn't perceived above seemed to be ring connectivity, better to run it
    # here then for someone to inquire about ring sizes and get 0 when it shouldn't be
    oechem.OEFindRingAtomsAndBonds(oemol)
    
    #print('Final Molecule: ', oechem.OEMolToSmiles(oemol))
    return oemol


def rdmol_from_oemol(oemol):
    """
    Creates an openeye molecule object that is identical to the input rdkit molecule
    """
    #print("Starting molecule: ", oechem.OEMolToSmiles(oemol))

    # start function
    rdmol = Chem.RWMol()

    # RDKit keeps bond order as a type instead using these values, I don't really understand 7, 
    # I took them from Shuzhe's example linked above
    _bondtypes = {1: Chem.BondType.SINGLE,
                  1.5: Chem.BondType.AROMATIC,
                  2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE,
                  4: Chem.BondType.QUADRUPLE,
                  5: Chem.BondType.QUINTUPLE,
                  6: Chem.BondType.HEXTUPLE,
                  7: Chem.BondType.ONEANDAHALF,}

    # atom map lets you find atoms again
    map_atoms = dict() # {oe_idx: rd_idx}
    for oea in oemol.GetAtoms():
        oe_idx = oea.GetIdx()
        rda = Chem.Atom(oea.GetAtomicNum())
        rda.SetFormalCharge(oea.GetFormalCharge())
        rda.SetIsAromatic(oea.IsAromatic())

        # unlike OE, RDK lets you set chirality directly
        cip = oechem.OEPerceiveCIPStereo(oemol, oea)
        if cip == oechem.OECIPAtomStereo_S:
            rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
        if cip == oechem.OECIPAtomStereo_R:
            rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

        map_atoms[oe_idx] = rdmol.AddAtom(rda)

    # As discussed above, setting bond stereochemistry requires neighboring bonds 
    # so we will store that information by atom index in this list 
    stereo_bonds = list()
    # stereo_bonds will have tuples with the form (rda1, rda2, rda3, rda4, is_cis)
    # where rda[n] is an atom index for a double bond of form 1-2=3-4 
    # and is_cis is a Boolean is True then onds 1-2 and 3-4 are cis to each other

    aro_bond = 0
    for oeb in oemol.GetBonds():
        # get neighboring rd atoms
        rd_a1 = map_atoms[oeb.GetBgnIdx()]
        rd_a2 = map_atoms[oeb.GetEndIdx()]

        # AddBond returns the total number of bonds, so addbond and then get it
        rdmol.AddBond(rd_a1, rd_a2)
        rdbond = rdmol.GetBondBetweenAtoms(rd_a1, rd_a2)

        # Assign bond type, which is based on order unless it is aromatic
        order = oeb.GetOrder()
        if oeb.IsAromatic():
            rdbond.SetBondType(_bondtypes[1.5])
            rdbond.SetIsAromatic(True)
        else:
            rdbond.SetBondType(_bondtypes[order])
            rdbond.SetIsAromatic(False)

        # If the bond has specified stereo add the required information to stereo_bonds
        if oeb.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
            # OpenEye determined stereo based on neighboring atoms so get two outside atoms
            n1 = [n for n in oeb.GetBgn().GetAtoms() if n != oeb.GetEnd()][0]
            n2 = [n for n in oeb.GetEnd().GetAtoms() if n != oeb.GetBgn()][0]

            rd_n1 = map_atoms[n1.GetIdx()]
            rd_n2 = map_atoms[n2.GetIdx()]
    
            stereo = oeb.GetStereo([n1,n2], oechem.OEBondStereo_CisTrans)
            if stereo == oechem.OEBondStereo_Cis:
                #print('cis')
                stereo_bonds.append((rd_n1, rd_a1, rd_a2, rd_n2, True))
            elif stereo == oechem.OEBondStereo_Trans:
                #print('trans')
                stereo_bonds.append((rd_n1, rd_a1, rd_a2, rd_n2, False))  
    
    # add bond stereochemistry:
    for (rda1, rda2, rda3, rda4, is_cis) in stereo_bonds:
        # get neighbor bonds
        bond1 = rdmol.GetBondBetweenAtoms(rda1, rda2)
        bond2 = rdmol.GetBondBetweenAtoms(rda3, rda4)
        
        # Since this is relative, the first bond always goes up
        # as explained above these names come from SMILES slashes so UP/UP is Trans and Up/Down is cis
        bond1.SetBondDir(Chem.BondDir.ENDUPRIGHT)
        if is_cis:
            bond2.SetBondDir(Chem.BondDir.ENDDOWNRIGHT)
        else:
            bond2.SetBondDir(Chem.BondDir.ENDUPRIGHT)
    
    # if oemol has coordinates (The dimension is non-zero)
    # add those coordinates to the rdmol
    if oechem.OEGetDimensionFromCoords(oemol) > 0:
        #print("oemol has 3D coords")
        conformer = Chem.Conformer()
        oecoords = oemol.GetCoords()
        for oe_idx, rd_idx in map_atoms.items():
            (x,y,z) = oecoords[oe_idx]
            conformer.SetAtomPosition(rd_idx, Geometry.Point3D(x,y,z))
        rdmol.AddConformer(conformer)    
    
    # Save the molecule title
    rdmol.SetProp("_Name", oemol.GetTitle())
    
    # Cleanup the rdmol
    # Note I copied UpdatePropertyCache and GetSSSR from Shuzhe's code to convert oemol to rdmol here:
    rdmol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(rdmol)
    # I added AssignStereochemistry which takes the directions of the bond set 
    # and assigns the stereochemistry tags on the double bonds
    Chem.AssignStereochemistry(rdmol, force=False) 
    
    #print("Final Molecule: ", Chem.MolToSmiles(Chem.RemoveHs(rdmol), isomericSmiles=True))
    return rdmol.GetMol()


def rocs_screening(query_list, reference):
    
    refmol = oemol_from_rdmol(deepcopy(reference))
    #prep = oeshape.OEOverlapPrep() # colors
    #prep.Prep(refmol) # colors
    overlay = oeshape.OEMultiRefOverlay()
    overlay.SetupRef(refmol)
    
    scores = []
    for query_mol in query_list:
        fitmol = oemol_from_rdmol(deepcopy(query_mol))
        #prep.Prep(fitmol)
        
        # Sort all scores according to highest tanimoto
        scoreiter = oeshape.OEBestOverlayScoreIter()
        oeshape.OESortOverlayScores(scoreiter, overlay.Overlay(fitmol), oeshape.OEHighestTanimoto())
        
        for score in scoreiter: # only 1 conformer, so only 1 score (iterates through 1 time)
            outmol = oechem.OEGraphMol(fitmol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx())))
            score.Transform(outmol)

        try:
            outmol_rdkit = rdmol_from_oemol(outmol)
            scores.append((score.GetTanimoto(), get_ROCS_mols(reference, outmol_rdkit).item()))
        
        except:
            scores.append((score.GetTanimoto(), None))
    
    return scores
    

if __name__ == '__main__':
    
    def logger(text, file = 'optimization_results_MOSES2/optimization_shape_screening_scores_MOSES2_log.txt', save = True):
        if save:
            with open(file, 'a') as f:
                f.write(text + '\n')
        else:
            print(text + '\n')
    
    filtered_database = pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database.pkl')
    filtered_database_mols = list(pd.read_pickle('data/MOSES2/MOSES2_training_val_filtered_database_artificial_mols.pkl').artificial_mols)
    filtered_database['rdkit_mol_cistrans_stereo'] = filtered_database_mols
    
    train_smiles_df = pd.read_csv('data/MOSES2/MOSES2_train_smiles_split.csv')
    train_smiles = set(train_smiles_df.SMILES_nostereo)
    train_db_mol = filtered_database.loc[filtered_database['SMILES_nostereo'].isin(train_smiles)].reset_index(drop = True)
    
    
    test_mol_df = pd.read_pickle('data/MOSES2/test_MOSES_filtered_artificial_mols.pkl')
    test_mols = list(test_mol_df.artificial_mol)
    
    reference_mols_index = [99300, 142337, 94211, 13059, 138951, 67478, 2775, 7994, 10770, 108203, 126430, 9126, 78600, 81366, 46087, 76561, 87747, 91918, 118822, 132656, 130062, 113584, 115006, 140953, 33351, 14473, 101938, 6686, 1200, 69153, 25628, 25659, 56430, 137033, 48156, 68289, 128739, 70016]
    reference_mols = [test_mols[i] for i in reference_mols_index]
    
    
    query_mols = list(train_db_mol.rdkit_mol_cistrans_stereo) # all of the training set
    
    scores_dict = {}
    
    for r in range(len(reference_mols)):
        
        reference = deepcopy(reference_mols[r])
        reference_index = reference_mols_index[r]
        
        N_chunks = 500
        num_per_chunk = len(query_mols) // N_chunks
        query_chunks = [query_mols[num_per_chunk*i:num_per_chunk*(i+1)] for i in range(math.ceil(len(query_mols) / num_per_chunk))]    
        logger('processing chunks...')
        chunk_scores = []
        pool = Pool(18)    
        for scores in tqdm(pool.imap(partial(rocs_screening, reference = reference), query_chunks), total = N_chunks):
            chunk_scores.append(scores)
        pool.close()
        pool.join()
        
        flattened_scores = [item for sublist in chunk_scores for item in sublist]
        
        scores_dict[reference_index] = flattened_scores
        
        logger(f'{r}: saving scores to dictionary...')
        with open('optimization_results_MOSES2/training_set_shape_screening_scores.pkl', 'wb') as f:
            pickle.dump(scores_dict, f)
    
    