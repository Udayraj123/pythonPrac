import tensorflow as tf
import numpy as np
import codecs
from Model import Model
import argparse

n_classes = 5
learning_rate = 0.5
emb_size = 200
batch_size = 10
_UNK = 0
num_hidden = 1
hidden_size = [100]
dropout = 0.5
window_size = 5
batch_size = 10
logs_path = '/tmp/model'
epochs = 2

parser = argparse.ArgumentParser()
parser.add_argument("file", help="path for word vector file")
parser.add_argument("model", help="name of model")
parser.add_argument("-a" ,"--add_vocab", help="path for vocab file", default="vocab_above5")
args = parser.parse_args()


def create_vocabulary(file, emb_size):
	vecs = np.array([np.zeros(emb_size)])
	mapping = {}
	f = codecs.open('data/'+file, 'rb', encoding='utf-8')
	idx = 1
	for line in f:
		split_line = line.split()
		if split_line[0] == 'UNK':
			vecs[0] = np.array([float(a) for a in split_line[1:-1]]).reshape(1,emb_size)
		else:
			mapping[split_line[0]] = idx
			idx += 1
			t_vecs = np.array([float(a) for a in split_line[1:-1]])
			vecs = np.concatenate((vecs, t_vecs.reshape(1,emb_size)), axis=0)
	# vecs = np.delete(vecs, (0), axis=0)
	return mapping, vecs

def add_vocabulary(file, mapping, embedding):
	f = codecs.open('data/'+file, 'rb', encoding='utf-8')
	for line in f:
		idx = len(mapping)
		if line.strip() not in mapping:
			mapping[line.strip()] = idx
			idx += 1
			embedding =  np.concatenate((embedding, np.random.rand(1,emb_size)), axis=0)
	return mapping, embedding

def get_mapping(file):
	mapping = {}
	f = codecs.open('data/'+file, 'rb', encoding='utf-8')
	idx = 0
	for line in f:
		mapping[line.strip()] = idx
		idx += 1
	return mapping	

def read_file(file, inp_map, out_map):
	X = []
	y = []
	f = codecs.open('data/'+file, 'rb', encoding='utf-8')
	for line in f:
		split_line = line.split()
		if len(split_line) > 0:
			inp_id = inp_map.get(split_line[0], _UNK)
			out_id = out_map.get(split_line[-1])
			X.append(inp_id)
			y.append(out_id)
	X = X[-2:] + X + X[:2]
	# y = y[-2:] + y + y[:2]
	final_X = [[X[i-2], X[i-1], X[i], X[i+1], X[i+2]] for i in xrange(2, len(X)-2)]
	# final_y = [[y[i-2], y[i-1], y[i], y[i+1], y[i+2]] for i in xrange(2, len(y)-2)]
	return final_X, y

def prepare_data(train_file, val_file, test_file, inp_map, out_map):
	train_X, train_y = read_file(train_file, inp_map, out_map)
	val_X, val_y = read_file(val_file, inp_map, out_map)
	test_X, test_y = read_file(test_file, inp_map, out_map)
	return train_X, train_y, val_X, val_y, test_X, test_y


keep_prob = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.int64 , [None])
inputs = tf.placeholder(tf.int64, [None, window_size])

inp_map, embedding = create_vocabulary(args.file, emb_size)
if args.add_vocab:
	inp_map, embedding = add_vocabulary(args.add_vocab, inp_map, embedding)
out_map = get_mapping('out_vocab')
train_X, train_y, val_X, val_y, test_X, test_y = prepare_data('eng.train', 'eng.testa', 'eng.testb', inp_map, out_map)
n_classes = len(out_map)
vocabulary_size = len(inp_map)
model = Model(num_hidden, hidden_size, window_size, emb_size, vocabulary_size, n_classes, inputs, labels, keep_prob, learning_rate, embedding)


# tf.summary.scalar("loss", model.loss)
# tf.summary.scalar("accuracy", model.accuracy)
# summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

saver = tf.train.Saver()

no_of_batches = int(len(train_X)/batch_size)
# writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
print("Starting Training")

for i in xrange(1, epochs+1):
	print ("Epoch: "+str(i))
	for j in xrange(no_of_batches):
		# _, summary = sess.run([model.optimize, summary_op], {inputs: train_X[j*batch_size : (j+1)*batch_size], labels: train_y[j*batch_size : (j+1)*batch_size], keep_prob: dropout})
		_ = sess.run(model.optimize, {inputs: train_X[j*batch_size : (j+1)*batch_size], labels: train_y[j*batch_size : (j+1)*batch_size], keep_prob: dropout})
		# writer.add_summary(summary, i * no_of_batches + j)
	acc_t, loss_t = sess.run([model.accuracy, model.loss], {inputs: train_X, labels: train_y, keep_prob: dropout})
	acc_v, loss_v = sess.run([model.accuracy, model.loss], {inputs: val_X, labels: val_y, keep_prob: dropout})
	print("Training- Accuracy: %f and Loss: %f") % (acc_t, loss_t)
	print("Validation- Accuracy: %f and Loss: %f") % (acc_v, loss_v)

print("Training Completed")
saver.save(sess, args.model+"/"+args.model)
sess.close()