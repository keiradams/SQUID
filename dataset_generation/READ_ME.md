This directory contains all the scripts necessary to create the datasets for training and evaluating SQUID. To create the datasets, all we require is a csv file of SMILES strings of molecules we wish to train on, as well as a separate csv file of SMILES strings of molecules we wish to test (evaluate) on. You can find the (relevant) outputs of these scripts in our downloadable data: https://figshare.com/s/3d2f8fd57d9a65fe237e

We use the train/test sets natively provided by MOSES.

TRAIN: data/MOSES2/train_MOSES.csv
TEST: data/MOSES2/test_MOSES.csv

To create all the data required to train and evaluate SQUID, run the following scripts (in the main directory) sequentially:

1. python MOSES2_training_val_dataset_generation.py
2. python combine_generation_data_MOSES2.py
3. python combine_scorer_data_MOSES2.py
4. python generate_artificial_mols_MOSES2.py
5. python MOSES2_training_val_arrays.py
6. python get_max_future_rocs_artificial_MOSES2.py {i} # for i = 0, 1, 2, ..., 15
7. python combine_maxFutureRocs_MOSES2.py
8. python create_test_mols_MOSES2.py

This data-preprocessing can be quite time and memory consuming, especially Step 6, when using a large dataset (>1M molecules). We ran each script on 1 node with 24 cpu cores (8 GB RAM each). We separately ran the 16 scripts on Step 6 independently and in parallel, on 16 different nodes (384 cores total). Each script in Step 6 takes approximately 3 days to complete.


Note that in total, the outputs of these scripts are quite large (50-100 GB).