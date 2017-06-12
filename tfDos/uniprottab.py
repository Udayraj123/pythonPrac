
"""
Benchmark link : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN

"""

import re
import os
import pickle
import pandas as pd
import numpy as np

def save_obj(obj, name ,overwrite=1):
	filename='data/'+ name + '.pkl';
	if(overwrite==1 and os.path.exists(name)):
		return [];
	with open(filename, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	filename='data/'+ name + '.pkl';
	# if(not os.path.exists(name)):
	# 	return [];
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

seq_records =pd.read_table("./data/uniprot-all.tab.gz", sep='\t')
i=1
num_recs=40
print(seq_records.describe())