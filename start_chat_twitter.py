import tensorflow as tf
import numpy as np
import seq2seq_wrapper

# preprocessed data
from datasets.twitter import data
import data_utils

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

import importlib
importlib.reload(seq2seq_wrapper)

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 256)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

sess = model.restore_last_session()

input_ = test_batch_gen.__next__()[0]
output = model.predict(sess, input_)
#print(output.shape)

#chat with user until user say bye----------------
while(True):
  temp = input_.T

  def c_s2i(ques, w2idx):
    return [w2idx.get(word, w2idx['unk']) for word in ques.split()]

  question = input("You : ")
  if question == "bye":
    print("Chatbot :  bye see you later :)")
    break
  ques = c_s2i(ques, metadata['w2idx'])
  question = question + [metadata['w2idx']['_']] * (20-len(ques))

  for x in range(20):
    temp[0,x] = question[x]

  predicted_answer = model.predict(sess, temp.T)

  raw_answer = data_utils.decode(sequence = predicted_answer[0], lookup=metadata['idx2w'], separator=' ').split(' ')

  final_answer = ' '.join(raw_answer)
  print("Chatbot : " + final_answer)
