# Building The Best ChatBot with Deep NLP

# Importing the libraries
import seq2seq_wrapper
import importlib
importlib.reload(seq2seq_wrapper)
import data_preprocessing
import data_utils_1
import data_utils_2


########## PART 1 - DATA PREPROCESSING ##########

# Importing the dataset
metadata, idx_q, idx_a = data_preprocessing.load_data(PATH = './')

# Splitting the dataset into the Training set and the Test set
(trainX, trainY), (testX, testY), (validX, validY) = data_utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = data_utils_2.get_metadata()


########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########

# Building the seq2seq model
model = seq2seq_wrapper.Seq2Seq(xseq_len = xseq_len,
                                yseq_len = yseq_len,
                                xvocab_size = xvocab_size,
                                yvocab_size = yvocab_size,
                                ckpt_path = './weights',
                                emb_dim = emb_dim,
                                num_layers = 3)

session = model.restore_last_session()

####### FREEZE THE MODEL ######################
save_path="./freeze_files/"
with session as sess:
    tf.train.write_graph(sess.graph_def, save_path, 'savegraph.pbtxt') #saving the model's tensorflow graph definition
    
import numpy as np
from tensorflow.python.tools import freeze_graph

# Freeze the graph
weight_path="./weights/"
save_path="./freeze_files/" #directory to model files
MODEL_NAME = 'Sample_model' #name of the model optional
input_graph_path = save_path+'savegraph.pbtxt'#complete path to the input graph
checkpoint_path = weight_path+'seq2seq_model.ckpt' #complete path to the model's checkpoint file
input_saver_def_path = ""
input_binary = False
#output node's name. Should match to that mentioned in your code
output_node_names = 'decoder/rdx_output'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = save_path+'frozen_model_'+MODEL_NAME+'.pb' # the name of .pb file you would like to give
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
