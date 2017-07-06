import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os
from nltk import ngrams
import pickle

print("Loading the data : ")
train_data = np.load('./data/cullpdb+profile_6133_filtered.npy')
# train_data = np.load('./data/cb513+profile_split1.npy')
train_rows = 5534
# train_rows = 514
print("Original shape : ", train_data.shape)

def save_obj(obj,filename,overwrite=1):
	if(not overwrite and os.path.exists(filename)):
		return
	with open(filename,'wb') as f:
		pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
		print("File saved to " + filename)

def load_obj(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		print("File loaded from " + filename)
		return obj

def read_glove_vec_files():
	file_path = './data/vectors_u.txt'
	file = open(file_path, 'r')
	word_to_glove = {}
	for line in file:
		line = line.split()	
		word = line[0]
		glove_vec = []
		for i in range(1, 101):
			glove_vec.append(float(line[i]))
		word_to_glove[word] = glove_vec
	# print(word_to_glove['L'])
	# print(word_to_glove['dummy'])
	# print(word_to_glove['X'])
	file.close()
	return word_to_glove

def corpus_creation():
	train_data_n = np.reshape(train_data, [-1, 57])
	print(train_data_n.shape, train_data_n.shape[0] == 700 * train_data.shape[0])
	amino_acids = train_data_n[:, 0:21]
	print(amino_acids.shape)
	no_of_amino_acids = np.sum(amino_acids, axis = 0)
	print(no_of_amino_acids)
	t_no_of_amino_acids = np.sum(no_of_amino_acids)
	print(t_no_of_amino_acids)
	no_seq = train_data_n[:, 21]
	t_no_of_no_seq = np.sum(no_seq)
	print(t_no_of_amino_acids, t_no_of_no_seq, t_no_of_amino_acids + t_no_of_no_seq)
	amino_acids_with_no_seq = train_data_n[:, 0:22]
	amino_acids_no = np.argmax(train_data_n, 1)
	no_to_am_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
	am_acids_name = []
	for i in range(amino_acids_with_no_seq.shape[0]):
		amino_acid_no = amino_acids_no[i].tolist()
		am_acids_name.append(no_to_am_acid[amino_acid_no])
	amino_acids_total = 0
	no_seq_total = 0
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			no_seq_total += 1
		else:
			amino_acids_total += 1
	print("amino_acids_total", amino_acids_total)
	print("no_seq_total", no_seq_total)

	seqs = {}
	for i in range(train_rows):
		seqs[i] = ""
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			continue
		else:
			seqs[i // 700] += am_acid_name

	total_len_of_all_seqs = 0
	for i in range(train_rows):
		total_len_of_all_seqs += len(seqs[i])

	print("Total len verfn results : ", total_len_of_all_seqs == amino_acids_total)

	def verify(seq_no):
		curr_rec = train_data[seq_no, :]
		amino_acids_no_cr = []
		for i in range(700):
			amino_acids_no_cr.append(np.argmax(curr_rec[i*57 : i*57 + 22], 0))
		curr_rec_seq=""
		for no in amino_acids_no_cr:
			if no == 21:
				break
			curr_rec_seq += no_to_am_acid[no]
		return curr_rec_seq == seqs[seq_no]

	verify(0)
	ans = True
	for i in range(train_rows):
		ans = ans and verify(i)
	print("Verification results : ", ans)

	def unigram_corpus_creation():
		tot_seq = " dummy" * 12
		for i in range(train_rows):
			seq = seqs[i]
			spaced_seq = ""
			# print(seq)
			for x in seq:
				spaced_seq += " " + x
			tot_seq += spaced_seq + " dummy" * 12
			# print(tot_seq)
		file_path = "./data/unigram_corpus"
		# print(tot_seq[:2000])
		# print(tot_seq[len(tot_seq)-2000:len(tot_seq)])
		if(os.path.exists(file_path)):
			return
		with open(file_path,'w') as f:
			f.write(tot_seq)

	def trigram_corpus_creation():
		tot_seq_0 = " dummy" * 12
		tot_seq_1 = ""
		tot_seq_2 = ""
		
		for i in range(train_rows):
			seq = seqs[i]
			seq = "$" + seq + "$"
			spaced_seq = ""
			for x in seq:
				spaced_seq += " " + x
			
			trigrams = ngrams(spaced_seq.split(), 3)
			tri_seq_list_0 = []
			tri_seq_list_1 = []
			tri_seq_list_2 = []

			counter_gram = 0
			for gram in trigrams:
				if(counter_gram%3 == 0):
					tri_seq_list_0.append(gram)
				if(counter_gram%3 == 1):
					tri_seq_list_1.append(gram)
				if(counter_gram%3 == 2):
					tri_seq_list_2.append(gram)
				counter_gram += 1

			tri_seq_str_0 = ""
			tri_seq_str_1 = ""
			tri_seq_str_2 = ""

			for gram in tri_seq_list_0:
				tri_seq_str_0 += " " + gram[0] + gram[1] + gram[2] 
			for gram in tri_seq_list_1:
				tri_seq_str_1 += " " + gram[0] + gram[1] + gram[2] 
			for gram in tri_seq_list_2:
				tri_seq_str_2 += " " + gram[0] + gram[1] + gram[2] 
			
			tot_seq_0 += tri_seq_str_0 + " dummy" * 12		
			tot_seq_1 += tri_seq_str_1 + " dummy" * 12		
			tot_seq_2 += tri_seq_str_2 + " dummy" * 12		
		tot_seq = tot_seq_0 + tot_seq_1 + tot_seq_2
		file_path = "./data/trigram_corpus"
		# print(tot_seq[:2000])
		# print(tot_seq[len(tot_seq)-2000:len(tot_seq)])
		if(os.path.exists(file_path)):
			return
		with open(file_path,'w') as f:
			f.write(tot_seq)

	# trigram_corpus_creation()
	# unigram_corpus_creation()
	for i in range(train_rows):
		print(len(seqs[i]))

def raw_data_to_mini_batches():
	train_data_n = np.reshape(train_data, [-1, 57])
	print(train_data_n.shape, train_data_n.shape[0] == 700 * train_data.shape[0])
	amino_acids = train_data_n[:, 0:21]
	amino_acids_seq_profile = train_data_n[:, 35:57]
	print(amino_acids.shape)
	no_of_amino_acids = np.sum(amino_acids, axis = 0)
	print(no_of_amino_acids)
	t_no_of_amino_acids = np.sum(no_of_amino_acids)
	print(t_no_of_amino_acids)
	no_seq = train_data_n[:, 21]
	t_no_of_no_seq = np.sum(no_seq)
	print(t_no_of_amino_acids, t_no_of_no_seq, t_no_of_amino_acids + t_no_of_no_seq)
	amino_acids_with_no_seq = train_data_n[:, 0:22]
	amino_acids_str_with_no_seq = train_data_n[:, 22:31]
	str_wise_sum = np.sum(amino_acids_str_with_no_seq, axis = 0)
	amino_acids_str_present = np.sum(str_wise_sum[:8])
	amino_acids_dum_present = np.sum(str_wise_sum[8])
	amino_acids_str_no = np.argmax(amino_acids_str_with_no_seq, 1)
	print("Str wise sum : ", str_wise_sum)
	print("Str present, padded data : ", amino_acids_str_present, amino_acids_dum_present, amino_acids_str_present + amino_acids_dum_present)
	amino_acids_no = np.argmax(train_data_n, 1)
	no_to_am_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
	am_acids_name = []
	for i in range(amino_acids_with_no_seq.shape[0]):
		amino_acid_no = amino_acids_no[i].tolist()
		am_acids_name.append(no_to_am_acid[amino_acid_no])
	amino_acids_total = 0
	no_seq_total = 0
	amino_acids_x = 0
	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			no_seq_total += 1
		else:
			if(am_acid_name == 'X'):
			  amino_acids_x += 1
			amino_acids_total += 1
	print("amino_acids_total", amino_acids_total)
	print("no_seq_total", no_seq_total)
	print("amino_acid_x_total", amino_acids_x)

	seqs = {}
	seq_pro = {}
	for i in range(train_rows):
		seqs[i] = ""
		seq_pro[i] = []

	for i in range(len(am_acids_name)):
		am_acid_name = am_acids_name[i]
		if(am_acid_name == 'NoSeq'):
			continue
		else:
			seqs[i // 700] += am_acid_name
			seq_pro[i // 700].append(amino_acids_seq_profile[i].tolist())

	total_len_of_all_seqs = 0
	for i in range(train_rows):
		total_len_of_all_seqs += len(seqs[i])

	print("Total len verfn results : ", total_len_of_all_seqs == amino_acids_total)

	seqs_in_vec = []
	masks = []
	ops = []
	seq_len = []
	zeros_list = [0] * len(seq_pro[0][0])
	word_to_glove = read_glove_vec_files()
	for i in range(train_rows):
		seq = seqs[i]
		temp_seq = []
		temp_ops = []
		temp_msk = []

		for j in range(50):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove["dummy"])
			glove_and_seq_pro_list.extend(zeros_list)
			temp_seq.append(glove_and_seq_pro_list)
			# temp_seq.append(word_to_glove["dummy"])
			temp_ops.append(-1)
			temp_msk.append(0)

		for j in range(len(seq)):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove[seq[j]])
			glove_and_seq_pro_list.extend(seq_pro[i][j])
			temp_seq.append(glove_and_seq_pro_list)
			temp_ops.append(amino_acids_str_no[i*700 + j])
			temp_msk.append(1)

		for j in range(750 - len(seq)):
			glove_and_seq_pro_list = []
			glove_and_seq_pro_list.extend(word_to_glove["dummy"])
			glove_and_seq_pro_list.extend(zeros_list)
			temp_seq.append(glove_and_seq_pro_list)
			temp_ops.append(-1)
			temp_msk.append(0)

		seqs_in_vec.append(temp_seq)
		ops.append(temp_ops)
		masks.append(temp_msk)
		seq_len.append(len(seq) + 100)

	ans = True
	count_masks_is_one = 0

	for j in range(train_rows):
		for i in range(800):
			if(masks[j][i] == 1):
				count_masks_is_one += 1
				ans = ans and (ops[j][i] != -1)
			else:
				ans = ans and (ops[j][i] == -1)

	for j in range(train_rows):
		ops_j = ops[j]
		for i in range(800):
			if(i<50 or i >= 50 + len(seqs[j])):
				ans = ans and (ops_j[i] == -1)
			else:
				ans = ans and (ops_j[i] != -1)

	ans = ans and ( count_masks_is_one == amino_acids_str_present)

	print("Verified the data inp, op and masks creation resuts : ", ans)

	batch_size = 256
	no_of_batches = train_rows // batch_size
	# train_rows // batch_size  = 43 for batch_size = 128
	# train_rows // batch_size  = 1106 for batch_size = 5
	# 0 - 42 batches with batch_size samples
	# 43 batch with 30 samples
	# 5504 + 30 samples in total
	mini_batch_data = {}
	print("Total number of batches : ", no_of_batches)
	for i in range(no_of_batches):
		temp = []
		temp.append(seqs_in_vec[i * batch_size : (i + 1) * batch_size ])
		temp.append(ops[i * batch_size : (i + 1) * batch_size ])
		temp.append(masks[i * batch_size : (i + 1) * batch_size ])
		temp.append(seq_len[i * batch_size : (i + 1) * batch_size ])
		mini_batch_data[i] = temp

	temp = []
	temp.append(seqs_in_vec[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	temp.append(ops[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	temp.append(masks[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	temp.append(seq_len[no_of_batches * batch_size : (no_of_batches + 1) * batch_size ])
	mini_batch_data[no_of_batches] = temp

	# total_samples = 0
	# for i in range(no_of_batches + 1):
	# 	total_samples += len(mini_batch_data[i][0]) 
		# print(len(mini_batch_data[i][0]))
	# print(total_samples)
	
	save_obj(mini_batch_data, './data/batch_wise_data_' + str(batch_size) + '.pkl')

# corpus_creation()
raw_data_to_mini_batches()


# word_to_glove =  read_glove_vec_files()
# print(word_to_glove.keys())
# print(len(word_to_glove.keys())) 23



 