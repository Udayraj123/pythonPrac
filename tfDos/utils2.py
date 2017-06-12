import numpy as np
import re
import pickle

def getFam(line):
	a = re.search(r'(.*)family',line);
	return str(a.group(1)).strip()

def getIDs(line):
	temp = [x for x in line.split() if '$_' in x]
	temp = [x.strip('$').strip('_') for x in temp]
	# a = re.search(r'\$_([\w\d]+)_\$',line);
	return temp

def saveFamilies():
	f = open("./data/similar", 'r')
	lines = str(f.read()).split('\n')
	ID_to_fam = {}
	IDs=[]
	totalIDs=0;
	for line in lines:
		if(line.strip()==""):
			continue;
		if("Highly divergent:" in line):
			continue;
		if("family" in line):
			IDs = np.array(IDs).flatten()
			totalIDs+=len(IDs)
			fam = getFam(line)
			for ID in IDs:
				ID_to_fam[ID] = fam
			IDs=[]
		else:
			lineIDs = getIDs(line)
			for x in lineIDs:
				IDs.append(x)

	print(totalIDs)

	save_obj(ID_to_fam,'ID_to_fam')


def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

saveFamilies()
ID_to_fam=load_obj('ID_to_fam');

print(ID_to_fam['Q2QGD7'])