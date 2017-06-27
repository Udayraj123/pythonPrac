
# coding: utf-8

# In[1]:


# Configuration -
num_ex = 500#000
famTHR = 200*num_ex/500000
min_after_dequeue = 1#10000*num_ex/500000
num_epochs=1
batch_size = 10
runonce=0
filepath='processed_uniprot.tfrecord'


# In[2]:


import re
import os
import string
import pickle
import tempfile
import numpy  as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plot

tf.logging.set_verbosity(tf.logging.ERROR)

def save_obj(obj, name ,overwrite=1):
    filename='data/'+ name + '.pkl';
    if(overwrite==1 and os.path.exists(name)):
        return [];
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    filename='data/'+ name + '.pkl';
    # if(not os.path.exists(name)):
    #   return [];
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

fam='Cross-reference (Pfam)'
seq = 'Sequence'

main =pd.read_table("./data/uniprot-all.tab.gz", sep='\t',nrows=num_ex)

main[fam]= main[fam].apply(lambda x:str(x).split(';')[0])
col = (main[fam]!='nan') & (main[fam].isnull()==False) & (main[seq].isnull()==False)
main = main[col]
alphafam = list(main[fam].unique())
num_classes = len(alphafam)+1 
alphabet =list(string.ascii_uppercase)
vocab_size = 1+len(alphabet)


# In[3]:


seq_records = main[[seq,fam,'Length']]
seq_records.columns=['seq','fam','length']
sample = seq_records#.sample(num_ex)


# In[4]:

#All the heavy jobs here - takes upto 92 seconds
seq_records = sample.copy()
seq_records['seq']= seq_records['seq'].apply(lambda seq : list(map(lambda x : 1+alphabet.index(x),list(seq))))
seq_records['fam']= seq_records['fam'].apply(lambda x : 1+alphafam.index(x))
#added 1+ for skipping 0 index
counts = seq_records.groupby(['fam']).size().reset_index(name='counts')


# In[5]:


#10801 -> 9254 families
counts = counts[ (counts['counts']>=famTHR) & (counts['counts'] < 3400) ]
#608 -> 555 families having count between 200 and 3400
filtered_fams = list(counts['fam'])
#323162 -> 302907 examples total satisfy it.
filtered_seqs = seq_records[ seq_records['fam'].isin(filtered_fams)]
filtered_seqs = filtered_seqs.sort_values('length')


# In[6]:


# takes 3 seconds !
gb = seq_records.groupby('fam')    
family_wise_db = [gb.get_group(x) for x in gb.groups]


# In[7]:


# seq_records = filtered_seqs
filtered_seqs.describe()


# In[8]:


"""
x = pd.DataFrame([[1],[2],[1],[3]])
x = x[(x==1) | (x==2)]
x.dropna()
"""
"""
SequenceData can be stored into TFRecords that is tf's own storage. 
It is easier to handle and reusable
& it follows the protocol buffer format already
"""
# efficient storage of the sequences
# per time step variable no of features.
"""
Padding = 
    static padding = using FIFOQueue
    dynamic padding = using tf.train.next_batch(..,dynamic_pad=True)
    Bucketting = tf.contrib.training.bucket_by_sequence_length(..,dynamic_pad=True)
    
Goal : Handle sequences of unknown lengths
tf.while_loop = dynamic loops and supports backprop grad descent
tf.while_loop(cond_fn,body_fn,loop_vars,)

But now to be able to process Slices of Tensors, use tf.TensorArray()
num_rows = matrix.shape[0]
ta = tf.TensorArray(tf.float32,size=num_rows)
loadedMatrix = ta.unstack(matrix)
read_row = loadedMatrix.read(row_idx)
loadedMatrix = loadedMatrix.write(row_idx,fn(read_row))
matrix = loadedMatrix.stack()

"""
    


# In[9]:



# Define how to parse the example - In the way you stored thru the recordWriter
context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64),
    "label": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
      """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def make2_example(sequence, label,sequence_length):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    ex.context.feature["label"].int64_list.value.append(label)
    # Feature lists for the two sequential features of our example
    fl_sequence = ex.feature_lists.feature_list["sequence"]
    for token in  sequence:
        fl_sequence.feature.add().int64_list.value.append(token)
    return ex

def make_example(sequence, label,length):
    feature = {
       'train/label': _int64_feature(label),
       'train/length': _int64_feature(length),
       'train/sequence': _bytes_feature(tf.compat.as_bytes(str(sequence)))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


# Write all examples into a TFRecords file
writer = tf.python_io.TFRecordWriter(filepath)

#     for sequence, label_sequence in zip(sequences, label_sequences):
for i,rec in filtered_seqs.iterrows():
    ex = make_example(rec.seq, rec.fam,rec.length)
    writer.write(ex.SerializeToString())

writer.close()
print("Wrote to {}".format(filepath))


# In[10]:


x =[{'length': 368,'a':'a'}, {'length': 337}, {'length': 338}, {'length': 338}]
# map(lambda x: (x['length'],x),x)
x[0].items()


# In[11]:


# a={'someKey':tf.Variable(5)}
# b=tf.Variable([5])
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# # print(tf.contrib.learn.run_n(a,n=4,feed_dict=None))
# # tf.Session().run([init,a])
# tf.Print(b[-1],[b]).eval()
num_classes


# In[12]:


class RnnForPfcModelOne:
#   @profile
    def __init__(self, 
        batch_size,
        num_classes = 549, 
        hidden_units=100,
        learning_rate=0.01):
        global vocab_size
        # batch_size * no_of_time_steps * vocab_size _/
        self.weights = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes], maxval=1))
        self.biases = tf.Variable(tf.random_uniform(shape=[num_classes]))
        self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
                                           forget_bias = 1.0,
                                           activation = tf.tanh)
        # self.len_data taken from feed_dict
        self.len_data = tf.placeholder(tf.uint8, [batch_size])
        # self.x_input taken from feed_dict
        self.x_input = tf.placeholder(tf.uint8, [None, None], name = 'x_ip') # batch_size * no_of_time_steps _/
        # self.x_input_o takes self.x_input
        self.x_input_o = tf.one_hot(indices = self.x_input, 
            depth = vocab_size,
            on_value = 1.0,
            off_value = 0.0,
            axis = -1)
        # self.outputs takes self.x_input_o & len_data
        self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
                                                      self.x_input_o,
                                                      sequence_length = self.len_data,
                                                      dtype = tf.float32)
        
        # outputs of shape batch_size * no_of_time_steps * vocab_size
        # output at time t i.e. the last output
        self.outputs_t = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
        # The single layer NN to classify - takes outputs_t
        self.y_predicted = tf.matmul(self.outputs_t, self.weights) + self.biases
        
        
        # self.y_input taken from feed_dict batch_size *1
        self.y_input = tf.placeholder(tf.uint8, [batch_size], name = 'y_ip')
        # self.y_input_o takes y_input
        self.y_input_o = tf.one_hot(indices = self.y_input, 
                                    depth = num_classes,
                                    on_value = 1.0,
                                    off_value = 0.0,
                                    axis = -1)
        
        # self.loss takes one hot y and y_predicted
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_input_o)
        #y_predicted and y_input_o shud be of same size = batch_size * num_classes
        # define optimizer and trainer
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.get_equal = tf.equal(tf.argmax(self.y_input_o, 1), tf.argmax(self.y_predicted, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))
        self.summary_writer = tf.summary.FileWriter('./data/graph/', graph = self.sess.graph)

#   @profile
    def predict(self, x, y, len_data):
        result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y, self.len_data:len_data})
        return result

#   @profile
    def optimize(self, x, y, len_data):
        self.sess.run(self.trainer, feed_dict={self.x_input: x, self.y_input: y, self.len_data:len_data})

#   @profile
    def cross_validate(self, x, y, len_data):
        result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y, self.len_data:len_data})
        return result

#   @profile
    def close_summary_writer(self):
        self.summary_writer.close()


# In[13]:


# tf.reset_default_graph()
#  READING DATA BACK- 
# A single serialized example


feature = {
            'train/sequence': tf.FixedLenFeature([], tf.string),
           'train/length': tf.FixedLenFeature([], tf.int64),
           'train/label': tf.FixedLenFeature([], tf.int64)
          }
# In[ ]:

def read_my_file_format(filequeue):
    reader = tf.TFRecordReader()
    _,serialized_example=reader.read(filequeue)
    features = tf.parse_single_example(serialized_example, features=feature)
    return features['train/sequence'],features['train/label'],features['train/length']
# In[ ]:

def input_pipeline(filepath, batch_size,pad_length,min_after_dequeue = 10000, num_epochs=None):
    filequeue = tf.train.string_input_producer([filepath],num_epochs=1)#,shuffle=True)
    sequence, label,length = read_my_file_format(filequeue)
    
    sequence = tf.decode_raw(sequence, tf.uint8)
        
    # Cast label data into int32
    label = tf.cast(label, tf.int32)
    length = tf.cast(length, tf.int32)
    sequence = tf.reshape(sequence, [pad_length])
    capacity = min_after_dequeue + 3 * batch_size

#  You can create the batch queue using tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    example_batch, label_batch = tf.train.shuffle_batch(
          [sequence, label], 
          # shapes=[[pad_length],[1]],
          batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


    
#Instead of running this tensor-
#         len_data = tf.contrib.learn.run_n(context_parsed, n=batch_size, feed_dict=None)
# we'll run a custom tensor now
# decode_x,decode_y,decode_len = read_and_decode_module(filepath,batch_size)


# In[14]:


if(not runonce):
    model = RnnForPfcModelOne(batch_size,num_classes=num_classes)
    runonce=1

#     output_dict: A dict mapping string names to tensors to run. Must all be from the same graph.
#     feed_dict: dict of input values to feed each run.
#     restore_checkpoint_path: A string containing the path to a checkpoint to restore.
#     n: Number of times to repeat.
"""
This comes  with np.array([1,2,3,[4,5,6]) or dtype is wrong
ValueError: setting an array element with a sequence.
So - padding was the culprit

"""
# data_train, data_test, data_cv = get_data(200)
# print(len(data_train), (len(data_test)), (len(data_cv)))
def pad(x,max_len):
    return np.lib.pad(x,(0,max_len - len(x)),'constant',constant_values=(-1,0))

# record_iterator = tf.python_io.tf_record_iterator(path=filepath)
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#     sequence = np.fromstring(example.features.feature['train/sequence'].bytes_list.value[0], dtype=np.uint8)
#     label = int(example.features.feature['train/label'].int64_list.value[0])
#     length = int(example.features.feature['train/length'].int64_list.value[0])
#     print(sequence,length,label)
    # debug=input()
        
pad_length=1000 # Get this from a file now
with tf.Session() as sess:
    batch_x, batch_y = input_pipeline(filepath,batch_size,pad_length,min_after_dequeue,num_epochs)
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(5):
        print('sess running')
        seq, lbl = sess.run([batch_x, batch_y ])
        print(seq,lbl)

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()



# for epoch in range(num_epochs):
#     no_of_batches = num_ex // batch_size
#     # queue for the data
#     for batch_no in range(no_of_batches):
# #         batch_x,batch_y,len_data = sess.run([decode_x,decode_y,decode_len])
#         # x_data = tf.contrib.learn.run_n(context_parsed, n=batch_size, feed_dict=None)
#         # y_data = tf.contrib.learn.run_n(sequence_parsed, n=batch_size, feed_dict=None)
#         print('sess running')
#         seq, lbl = model.sess.run([batch_x, batch_y ])
#         print(seq,lbl)
#         debug=input()
#         model.optimize(batch_x,batch_y,len_data)
#         accuracy_known = model.cross_validate(batch_x,batch_y,len_data)
#         print("Iteration number, batch number : ", epoch, batch_no," Training data accuracy : ", accuracy_known)

# model.close_summary_writer()
