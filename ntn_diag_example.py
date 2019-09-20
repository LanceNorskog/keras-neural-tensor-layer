import math
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import hiddenlayer as hl
import hiddenlayer.transforms as ht

from google.colab import files

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from ntn_diag import NeuralTensorDiagLayer

def get_data(num_samples=100000):
  digits = load_digits()
  num_digits = digits.data.shape[0]
  # create pairs of all images VS all images
  k = 0
  pair1 = []
  pair2 = []
  ltall = []
  for i in range(num_digits):
    for j in range(num_digits):
        if digits.target[i] < 5 and digits.target[j] > 4:
            pair1.append(i)
            pair2.append(j)
    
  pair1 = np.asarray(pair1, dtype='int32')
  pair2 = np.asarray(pair2, dtype='int32')
  num_pairs = len(pair1)
  print('total pairs: ', num_pairs)  
  print('total true: ', np.sum(digits.target[pair1] * 2 == digits.target[pair2]))

# pick a random subset of image pairs
  subset = np.arange(num_pairs, dtype='int32')
  np.random.shuffle(subset)
  if num_samples < num_pairs:
    num_pairs = num_samples
    subset = subset[0:num_samples]
  pair1 = pair1[subset]
  pair2 = pair2[subset]

  times2 = np.asarray(digits.target[pair1] * 2 == digits.target[pair2], dtype='int32')
  print('total true: ', np.sum(times2))
  times2 = np.reshape(times2, (num_pairs, 1))
    
  print(pair1[0:200])
  print(pair2[0:200])
  #print(eqall[0:200])
 
  return digits.data[pair1] / 16, digits.data[pair2] / 16, times2

X1_train,X2_train,Y_train = get_data(num_samples=20000)

squish=16
squish2=8

if True:
  input1 = Input(shape=(64,), dtype='float32')
  input2 = Input(shape=(64,), dtype='float32')
  i1 = Dense(units=squish, activation='relu')(input1)
  i2 = Dense(units=squish, activation='relu')(input2)
  # btp = NeuralTensorDiagLayer(output_dim=squish2)([i1, i2])
  btp = NeuralTensorDiagLayer(output_dim=squish2)([i1, i2])

  p = Dense(units=1, activation='sigmoid')(btp)
  model = Model(inputs=[input1, input2], outputs=[p])

  adam = Adam()
  model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
  model.summary()

#!ls -l /tmp/file*.pdf
#files.download('/tmp/filediag.pdf')

history = model.fit([X1_train, X2_train], Y_train, 
            validation_split=0.2,
            callbacks = [EarlyStopping(patience=5)],
            epochs=10, batch_size=16, verbose=2)

plt.figure()
metric_names = ['loss', 'binary_accuracy']
if history != None:
  # summarize history for accuracy
  for m in metric_names:
      #plt.plot(history.history[m])
      plt.plot(history.history['val_' + m])
  plt.title('model accuracy')
  plt.xlabel('epoch')
  sname = []
  for m in metric_names:
      sname.append('{}={:01.3f}'.format(m, history.history['val_' + m][-1]))
  plt.legend(sname, loc='center right')
  plt.show()
