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
                                ckpt_path = './weights/',
                                emb_dim = emb_dim,
                                num_layers = 3)
                                
                                
########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########

val_batch_gen = data_utils_1.rand_batch_gen(validX, validY, 256)
test_batch_gen = data_utils_1.rand_batch_gen(testX, testY, 256)
train_batch_gen = data_utils_1.rand_batch_gen(trainX, trainY, batch_size)

sess = model.train(train_batch_gen, val_batch_gen)

#input_ = test_batch_gen.__next__()[0]
#output = model.predict(sess, input_)
#print(output.shape)
