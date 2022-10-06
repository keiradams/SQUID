import pandas as pd
import numpy as np

print('reading computing rocs')
rocs_0 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_0.npy')
rocs_1 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_1.npy')
rocs_2 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_2.npy')
rocs_3 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_3.npy')
rocs_4 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_4.npy')
rocs_5 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_5.npy')
rocs_6 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_6.npy')
rocs_7 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_7.npy')
rocs_8 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_8.npy')
rocs_9 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_9.npy')
rocs_10 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_10.npy')
rocs_11 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_11.npy')
rocs_12 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_12.npy')
rocs_13 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_13.npy')
rocs_14 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_14.npy')
rocs_15 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_rocs_15.npy')

print('reading evaluated dihedrals')
evaluated_dihedrals_0 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_0.npy')
evaluated_dihedrals_1 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_1.npy')
evaluated_dihedrals_2 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_2.npy')
evaluated_dihedrals_3 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_3.npy')
evaluated_dihedrals_4 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_4.npy')
evaluated_dihedrals_5 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_5.npy')
evaluated_dihedrals_6 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_6.npy')
evaluated_dihedrals_7 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_7.npy')
evaluated_dihedrals_8 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_8.npy')
evaluated_dihedrals_9 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_9.npy')
evaluated_dihedrals_10 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_10.npy')
evaluated_dihedrals_11 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_11.npy')
evaluated_dihedrals_12 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_12.npy')
evaluated_dihedrals_13 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_13.npy')
evaluated_dihedrals_14 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_14.npy')
evaluated_dihedrals_15 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals_15.npy')

print('reading evaluated indices')
evaluated_indices_0 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_0.npy')
evaluated_indices_1 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_1.npy')
evaluated_indices_2 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_2.npy')
evaluated_indices_3 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_3.npy')
evaluated_indices_4 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_4.npy')
evaluated_indices_5 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_5.npy')
evaluated_indices_6 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_6.npy')
evaluated_indices_7 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_7.npy')
evaluated_indices_8 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_8.npy')
evaluated_indices_9 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_9.npy')
evaluated_indices_10 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_10.npy')
evaluated_indices_11 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_11.npy')
evaluated_indices_12 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_12.npy')
evaluated_indices_13 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_13.npy')
evaluated_indices_14 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_14.npy')
evaluated_indices_15 = np.load('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices_15.npy')

max_rocs = np.concatenate([
    rocs_0, 
    rocs_1, 
    rocs_2, 
    rocs_3, 
    rocs_4, 
    rocs_5, 
    rocs_6, 
    rocs_7,
    rocs_8,
    rocs_9, 
    rocs_10,
    rocs_11,
    rocs_12,
    rocs_13,
    rocs_14,
    rocs_15,  
], axis = 0)

evaluated_dihedrals = np.concatenate([
    evaluated_dihedrals_0, 
    evaluated_dihedrals_1, 
    evaluated_dihedrals_2, 
    evaluated_dihedrals_3, 
    evaluated_dihedrals_4, 
    evaluated_dihedrals_5, 
    evaluated_dihedrals_6, 
    evaluated_dihedrals_7,
    evaluated_dihedrals_8, 
    evaluated_dihedrals_9, 
    evaluated_dihedrals_10, 
    evaluated_dihedrals_11, 
    evaluated_dihedrals_12, 
    evaluated_dihedrals_13, 
    evaluated_dihedrals_14, 
    evaluated_dihedrals_15,
], axis = 0)

evaluated_indices = np.concatenate([
    evaluated_indices_0, 
    evaluated_indices_1, 
    evaluated_indices_2, 
    evaluated_indices_3, 
    evaluated_indices_4, 
    evaluated_indices_5, 
    evaluated_indices_6, 
    evaluated_indices_7,
    evaluated_indices_8, 
    evaluated_indices_9, 
    evaluated_indices_10, 
    evaluated_indices_11, 
    evaluated_indices_12, 
    evaluated_indices_13, 
    evaluated_indices_14, 
    evaluated_indices_15,
], axis = 0)

print('saving final arrays')
np.save('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_indices.npy', evaluated_indices)
np.save('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/evaluated_dihedrals.npy', evaluated_dihedrals)
np.save('data/MOSES2/max_future_rocs_data_artificial_alpha_2_0/computed_max_future_rocs.npy', max_rocs)

print('is evaluated_indices sorted?', np.all(evaluated_indices[:-1] <= evaluated_indices[1:]))

print('done')

