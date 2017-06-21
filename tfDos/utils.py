import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle

data = pd.read_table('./../data/uniprot-all.tab', sep = '\t')
data = data.dropna(axis = 0, how = 'any')

# fig, ax = plt.subplots()
# data['Protein families'].value_counts().plot(ax=ax, kind='bar')
# # plt.show()
# fig.savefig('./../data/family_freq')

data_np = data.as_matrix()
print("Data loaded and NaN values dropped, shape : ", data_np.shape)

def save_familywise_db(min_no_of_seq = 200):
	families = []
	for i in range(data_np.shape[0]):
		families.append(data_np[i, 3])
	families_count = Counter(families)
	
	no_of_families = 0
	families_included = []
	for k in families_count.keys():
		if(families_count[k] >= min_no_of_seq):
			no_of_families += 1
			families_included.append(k)
	# store the entire data family-wise
	# this would help to divide data 
	# into three parts with stratification

	db_ = {}
	for fam in families_included:
		db_[fam] = []

	for i in range(data_np.shape[0]):
		if(data_np[i, 3] in families_included):
			temp = [data_np[i, 0], data_np[i, 2], data_np[i, 3]]
			db_[data_np[i, 3]].append(temp)

	file_path = './../data/db_' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(db_, output)
		output.close()

	print(no_of_families)

def map_creator():
	amino_acid_map = {}
	amino_acid_map['A'] = 1
	amino_acid_map['C'] = 2
	amino_acid_map['D'] = 3 # aspartic acid
	amino_acid_map['E'] = 4
	amino_acid_map['F'] = 5
	amino_acid_map['G'] = 6
	amino_acid_map['H'] = 7
	amino_acid_map['I'] = 8
	amino_acid_map['K'] = 9
	amino_acid_map['L'] = 10
	amino_acid_map['M'] = 11
	amino_acid_map['N'] = 12
	amino_acid_map['P'] = 13
	amino_acid_map['Q'] = 14
	amino_acid_map['R'] = 15
	amino_acid_map['S'] = 16
	amino_acid_map['T'] = 17
	amino_acid_map['U'] = 18 # Q9Z0J5 - confused with v ?
	amino_acid_map['V'] = 18
	amino_acid_map['W'] = 19
	amino_acid_map['Y'] = 20
	amino_acid_map['X'] = 21 # Q9MVL6 - undetermined
	amino_acid_map['B'] = 22 # asparagine/aspartic acid
	amino_acid_map['Z'] = 23 # glutamine/glutamic acid P01340

	families = []
	for i in range(data_np.shape[0]):
		families.append(data_np[i, 3])
	families_count = Counter(families)

	families_map = {}
	counter = 0
	
	for k, v in families_count.most_common():
		counter += 1
		families_map[k] = counter
	
	"""
	Class-II aminoacyl-tRNA synthetase family 3729
	3729 1
	RRF family 764
	764 87
	TGF-beta family 213
	213 510
	"""

	file_path = './../data/amino_acid_map' +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(amino_acid_map, output)
		output.close()

	file_path = './../data/families_map' +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(families_map, output)
		output.close()

	

# Ran these once, so files are saved 
save_familywise_db()
save_familywise_db(100)
save_familywise_db(50)

map_creator()
