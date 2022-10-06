

https://user-images.githubusercontent.com/52709065/194365060-2179f96b-1512-4bbe-b1a8-d4b25f4951a5.mov

This repository contains the necessary scripts to train and evaluate SQUID from:

"Equivariant Shape-Conditioned Generation of 3D Molecules for Ligand-Based Drug Design"

Before running any scripts, please download the necessary data (>50 GB) from:

https://figshare.com/s/3d2f8fd57d9a65fe237e

This data includes:

- The original train/test splits from MOSES
- The filtered train/val/test splits, which remove molecules with fragments not included in our fragment library
- The fragment library
- The RDKit-generated 3D conformers (both with fixed and relaxed bonding geometries) for all data splits, including the test-set molecules used in the experiments
- The pre-processed training data required to train both the graph generator and the rotatable bond scorer.

You can find the scripts used to generate all these data, along with further instructions, in the directory dataset_generation/

The downloaded data also includes the SQUID-generated 3D molecules for the shape-conditioned generation of chemically diverse molecules experiment, along with the encoded target molecules. This is for user convenience, as re-running this experiment requires access to an OpenEye license.


You will also need to create a new Python (conda) environment with the dependencies listed in environment.yml. The core dependencies needed to run the scripts and notebooks in this repository are as follows:

- notebook (6.4.11) (for running the Jupter Notebook demonstration)
- python (3.10.4)
- cudatoolkit (11.3.1)
- networkx (2.7.1)
- numpy (1.22.3)
- torch (1.11.0) with (cuda 11.3)
- torch-geometric (2.0.4)
- torch-scatter (2.0.9)
- torch-sparse (0.6.13)
- torchaudio (0.11.0)
- torchvision (0.12.0)
- pandas (1.4.2)
- scipy (1.8.1)
- tqdm(4.64.0)
- rdkit (2022.03.2)
- openeye-toolkits (2022.1.1)

To use the openeye toolkits (for running SE(3)-alignments and computing (aligned) shape similarities), you will need access to an OpenEye license. Please see https://www.eyesopen.com/academic-licensing for details. 


This directory is organized as follows:

- dataset_generation/ contains scripts to process user-provided csv files of SMILES strings into 3D conformers (for evaluations) and training data
- models/ contains the implementation of SQUID
- utils/ contains support functions used across training and generation
- trained_models/ contains trained models for the graph-generator and the rotatable bond scorer
- MO_virtual_screening/ contains scripts used for virtual screening (VS) in the Shape-Constrained Molecular Optimization experiment

- train_graph_generator.py and train_scorer.py contain scripts to train the graph generator and the scorer, respectively.
- shape_conditioned_generation_evaluations.py contains the script to evaluate SQUID in the Shape-Conditioned Generation of Chemically Diverse Molecules experiment.
- shape_conditioned_generation_dataset_baseline.py contains the script to compute the dataset baseline for the Shape-Conditioned Generation of Chemically Diverse Molecules experiment.
- shape_constrained_optimization_evaluations.py contains the script to run the genetic algorithm for shape-constrained molecular optimization with SQUID.


Demonstration:

RUN_ME.ipynb provides a lightweight, interactive demonstration of how we can easily use SQUID to generate chemically diverse molecular analogues with high shape similarity to an encoded molecule.


(Re)Training:

After downloading the training data, you can train the graph generator and the scorer (with their default settings, on a gpu) by running:

python train_graph_generator.py

python train_scorer.py


Experiments:

You can re-generate the SQUID-generated molecules analyzed in the Shape-Conditioned Generation of Chemically Diverse Molecules experiment by running:

python shape_conditioned_generation_evaluations.py {experiment_name} {lambda_interp} {stop_threshold}

For instance, to generate 50 samples for 1000 target shapes in the test set using $\lambda = 1.0$ (the prior), as performed in the paper, run:

python shape_conditioned_generation_evaluations.py lambda10 1.0 0.01

This will create pickle files containing the generated molecules and the encoded molecules with the target shapes. We include these files in paper_results/, which can be downloaded along with the training data. These generated molecules can then be filtered by tanimoto similarity to the target and sampled to generate the histograms of shape similarity (vs. chemical similarity) in Figure 3 of our submission.


You can run the Shape-Constrained Molecular Optimization experiment by running:

python shape_constrained_optimization_evaluations.py {experiment_name} {objective} {mol_index}

The objective is one of GSK3B, JNK3, Osimertinib_MPO, Sitagliptin_MPO, Celecoxib_Rediscovery, or Thiothixene_Rediscovery. The mol_index corresponds to the index of a seed molecule in the test set. In our paper, we use different molecules for each objective. In particular, we use:

GSK3B: (99300, 142337, 94211, 13059, 138951, 67478, 128739, 70016)
JNK3: (2775, 7994, 10770, 108203, 126430, 9126, 128739, 70016)
Osimertinib_MPO: (78600, 81366, 46087, 76561, 87747, 91918, 128739, 70016)
Sitagliptin_MPO: (118822, 132656, 130062, 113584, 115006, 140953, 128739, 70016)
Celecoxib_Rediscovery: (33351, 14473, 101938, 6686, 1200, 69153, 128739, 70016)
Thiothixene_Rediscovery: (25628, 25659, 56430, 137033, 48156, 68289, 128739, 70016)

Hence, you can optimize GSK3B while restricting the shape similarity to moleule 99300 by running:

python shape_constrained_optimization_evaluations.py GSK3B_99300 GSK3B 99300

MO_virtual_screening/ contains the scripts used for virtual screening (VS) baseline, which includes screening for each objective and for shape similarity to each of the seed molecules with the target shape.


Finally, please note that this repository will be updated throughout and after the review period (as permitted by ICLR) to improve the usability of SQUID. After decisions are announced and papers de-anonymized, a public open-source Github repository will be shared that contains updated training/evaluation scripts, demonstrations, and interactive tutorials.

