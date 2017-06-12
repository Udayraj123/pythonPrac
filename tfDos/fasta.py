
"""
Benchmark link : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN

"""

import re
import os
import pickle
import numpy as np
from Bio import SeqIO
# from pyquery import PyQuery as pq
# import urllib
# from lxml import html
# from lxml.cssselect import CSSSelector
# def scrapeFame(ID):
	# url='http://pfam.xfam.org/protein/'+ID
	# result = urllib.urlopen(url).read()
	# h= html.fromstring(result)
	# selector = CSSSelector("a[href^='http://pfam.xfam.org/family/']")
	# families = [e.text_content() for e in selector(h)]
	# return families

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

ID_to_fam=load_obj('ID_to_fam');

counter=0
def getFam(ID):
	global counter
	try:
		fam = ID_to_fam[ID]
		# print(fam)
	except KeyError:
		counter+=1
		print(ID)
		fam=None;
	return fam


seq_records =SeqIO.parse("./data/uniprot_sprot.fasta", "fasta")
temp2 = []
for seq in seq_records:
	temp2.append(seq.id.split('|')[1])

np_records = []
word = r'([\w\d'+re.escape("\"?}{\\^.'#:,()/-+[]<>*;")+']+)'; #r'([^=\t ])'
wordsp = r'([\w\d\t '+re.escape("\"}{\\^?.'#:,()/-+[]<>*;")+']+)'; # r'([^=])'
descRegX= (word+r'\|'+word+r'\|'+word+r' '+wordsp+r'OS='+wordsp+r' GN='+wordsp+r' PE='+word+r' SV='+word);
descRegX2= (word+r'\|'+word+r'\|'+word+r' '+wordsp+r'OS='+wordsp+r' PE='+word+r' SV='+word);
# sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) GN=FV3-001R PE=4 SV=1
#	sp|UniqID| ??  		Protein Name 					OS=.. 							 GN=.. 			PE=.. 	SV=..s
# 'sp|Q9SP07|1433_LILLO 14-3-3-like protein OS=Lilium longiflorum PE=2 SV=1'
# sp|Q5PRD0|143BA_DANRE 14-3-3 protein beta/alpha-A OS=Danio rerio GN=ywhaba PE=2 SV=2
num_recs=40000000
start=0
i=start
seq1_records =SeqIO.parse("./data/uniprot_sprot.fasta", "fasta")

for req in seq1_records:
	if(i==start+num_recs):break;
	i+=1
	description = req.description
	matches=[]
	a=re.search(descRegX,description)
	if(a):
		p=7
		GN=a.group(6)
	else: 
		a=re.search(descRegX2,description)
		p=6
		GN=None
		if not a : 
			print(i,'DNE',description)
			debug = input()
			continue
	ID=a.group(2)
	fam = getFam(ID)
	if(fam):
		np_records.append({
			'id' : ID,
	        'family' : fam,
			'seq' : req.seq,
			'ProteinName' : a.group(4),
			'OS' : a.group(5),
			'GeneName' : GN,
			'PE' : a.group(p),
			'SV' : a.group(p+1),
			})
	else:
		if(ID in temp2):
			print("Bug found here  :  ")
		else:
			print("ok")


# np_records = np.array(np_records)#
# save_obj(np_records,'np_records')
print(counter)
print(len(np_records))
print(counter + len(np_records))

