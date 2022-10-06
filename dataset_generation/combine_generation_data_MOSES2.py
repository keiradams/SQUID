import pandas as pd
import numpy as np

df1 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_1_reduced.pkl')
print('1')
df2 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_2_reduced.pkl')
print('2')
df3 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_3_reduced.pkl')
print('3')
df4 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_4_reduced.pkl')
print('4')
df5 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_5_reduced.pkl')
print('5')
df6 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_6_reduced.pkl')
print('6')
df7 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_7_reduced.pkl')
print('7')
df8 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_8_reduced.pkl')
print('8')
df9 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_9_reduced.pkl')
print('9')
df10 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_10_reduced.pkl')
print('10')
df11 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_11_reduced.pkl')
print('11')
df12 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_12_reduced.pkl')
print('12')
df13 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_13_reduced.pkl')
print('13')
df14 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_14_reduced.pkl')
print('14')
df15 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_15_reduced.pkl')
print('15')
df16 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_16_reduced.pkl')
print('16')
df17 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_17_reduced.pkl')
print('17')
df18 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_18_reduced.pkl')
print('18')
df19 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_19_reduced.pkl')
print('19')
df20 = pd.read_pickle('data/MOSES2/uncombined_databases/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_20_reduced.pkl')
print('20')

df_combined = pd.concat([
    df1, 
    df2, 
    df3,
    df4,
    df5,
    df6,
    df7,
    df8,
    df9,
    df10,
    df11,
    df12,
    df13,
    df14,
    df15,
    df16,
    df17,
    df18,
    df19,
    df20,
]).reset_index(drop = True)

print(df_combined.info())

df_combined.to_pickle('data/MOSES2/MOSES2_training_val_canonical_terminalSeeds_unmerged_graph_construction_database_all_reduced.pkl')
