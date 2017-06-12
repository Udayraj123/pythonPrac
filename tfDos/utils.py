from Bio import SeqIO
import requests
from bs4 import BeautifulSoup
import pickle

def get_seq_identifier():
	seq_identifier = {}
	data = SeqIO.parse("./data/uniprot_sprot.fasta", "fasta")
	for seq_record in data :
		unique_identifier = seq_record.id.split('|')[1]
		seq_identifier[unique_identifier] = str(seq_record.seq)
	return seq_identifier
		
def get_all_identifiers():
	seq_identifier = get_seq_identifier()
	identifiers = list(seq_identifier.keys())
	return identifiers
# not found - Q197F7
def get_family_from_identifiers():
	
	# Web-scraping based approach 
	# items = get_all_identifiers()
	# items = ["Q197F7", "B2JFE1"]
	# family = {}
	# for i in range(len(items)):
	# 	unique_identifier = items[i]
	# 	url = "http://pfam.xfam.org/protein/" + unique_identifier
	# 	page = requests.get(url)
	# 	soup = BeautifulSoup(page.content, 'html.parser')
	# 	html = list(soup.children)[2]
	# 	temp = html.find_all('table', class_='resultTable details', id="imageKey")
	# 	temp = str(temp)
	# 	temp = temp.split('<')
	# 	find = "http://pfam.xfam.org/family"
	# 	family[unique_identifier] = "not_scraped"
	# 	for item in temp:
	# 		if(find in item):
	# 			item = item.split('>')
	# 			family_of_item = item[-1]
	# 			print(unique_identifier, family_of_item)
	# 			family[unique_identifier] = family_of_item

	# for item in items:
	# 	print(item, " : ", family[item])

	# output = open('identifier_to_family.txt', 'ab+')
	# pickle.dump(family, output)
	# output.close()

	# try to get data from file similar
	f = open("./data/similar", 'r')
	temp = str(f.read())
	temp = temp.split("II. Families")[1].split('---------------')[0].split('\n')
	temp = [x for x in temp if x]
	family_tree = {}
	curr_family= " "

	for i in range(len(temp)):
		if("family" in temp[i]):
			curr_family = temp[i]
			family_tree[curr_family] = []
		else:
			family_tree[curr_family].append(temp[i])
	
	for k in family_tree.keys():
		items = family_tree[k]
		s = ""
		for item in items:
			s += item
		temp = s.replace('<a', '$').replace('a>', '$').split('$')
		temp = [x for x in temp if "uniprot" in x]
		temp = [x.split('\'')[1].split('/')[2] for x in temp]
		family_tree[k] = temp

	identifier_to_family_dbase = {}
	for k in family_tree.keys():
		for identifier in family_tree[k]:
			identifier_to_family_dbase[identifier] = k

	items = get_all_identifiers()
	counter = 0
	i = 0

	identifier_to_family = {}
	for k in identifier_to_family_dbase.keys():
		i += 1
		if(i % 1000 == 0):
			print("Done for  : ", i)
			print("Found for : ", counter)
		if(k in items):
			identifier_to_family[k] = identifier_to_family_dbase[k]
			counter += 1
	
	output = open('identifier_to_family.txt', 'ab+')
	pickle.dump(identifier_to_family, output)
	output.close()
		
	print(counter)


get_family_from_identifiers()

