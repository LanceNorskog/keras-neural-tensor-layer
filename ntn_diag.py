#!/usr/bin/python

import scipy.stats as stats

from keras import backend as K
from keras.engine.topology import Layer

class NeuralTensorDiagLayer(Layer):
  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(NeuralTensorDiagLayer, self).__init__(**kwargs)


  def build(self, input_shape):
    mean = 0.0
    std = 1.0
    # W : k*d
    k = self.output_dim
    d = self.input_dim
    initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d))
    initial_V_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
    self.W = K.variable(initial_W_values, name='W')
    self.V = K.variable(initial_V_values, name='V')
    self.b = K.zeros((self.input_dim,), name='b')
    self.trainable_weights = [self.W, self.V, self.b]


  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) != 2:
      raise Exception('NTNDiagLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    k = self.output_dim
    print('inputs: ', [e1,e2])
    feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
    print('ff: ', feed_forward_product)
    diag_tensor_products = [] 
    for i in range(k):
      diag_tensor_products.append((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
    print('diag.shape: ', K.shape(diag_tensor_products[0]))
    result = K.tanh(K.reshape(K.concatenate(diag_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
    print('result: ', result)
    return result


  def compute_output_shape(self, input_shape):
    # print (input_shape)
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)