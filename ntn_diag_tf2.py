
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
import scipy.stats as stats
import numpy as np

class NeuralTensorDiagLayer(Layer):
  def __init__(self, output_dim, activation=K.tanh, collector=K.mean, feedforward=True, bias=True, **kwargs):
    self.output_dim = output_dim #k
    self.activation = activation
    self.collector = collector
    self.feedforward = feedforward
    self.bias = bias
    super(NeuralTensorDiagLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    assert input_shape[0][-1] == input_shape[1][-1]
    mean = 0.0
    std = 1.0
    # W : k*d
    k = self.output_dim
    d = input_shape[0][-1]
    self.input_dim = d

    #w_init = tf.initializers.truncated_normal(mean=mean, stddev=2*std)
    #v_init = tf.initializers.truncated_normal(mean=mean, stddev=2*std)
    w_init = tf.initializers.glorot_uniform
    v_init = tf.initializers.glorot_uniform
    b_init = tf.initializers.zeros
    self.W = self.add_weight(shape=(k, d), 
                             initializer=w_init,
                             trainable=True,
                             name='W')
    self.V = None
    self.B = None
    if self.feedforward:
        self.V = self.add_weight(shape=(2*d, k), 
                             initializer=v_init,
                             trainable=True,
                             name='V')
    if self.bias:
        self.b = self.add_weight(shape=(k), initializer=b_init, trainable=True, name='b')


  def call2(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) != 2:
      raise Exception('NTNDiagLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    k = self.output_dim
    if self.feedforward:
        feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
    diag_tensor_products = [] 
    for i in range(k):
      diag_tensor_products.append(self.collector(e2 * (e1 * self.W[i])))
    stacked = K.stack(diag_tensor_products)
    if self.feedforward and self.bias:
        result = stacked + feed_forward_product + self.b
    elif self.feedforward:
        result = stacked + feed_forward_product  
    elif self.bias:
        result = stacked + self.b
    else:
        result = stacked
    if self.activation:
        result = self.activation(result)
        
    return result

  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) != 2:
      raise Exception('NTNDiagLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    batch_size = inputs[0].shape[0]
    input_size = inputs[0].shape[-1]
    print('batch_size, input_size:', batch_size, input_size)
    batch_size = K.shape(inputs[0])[0]
    input_size = K.shape(inputs[0])[-1]
    print('batch_size, input_size:', batch_size, input_size)
    k = self.output_dim
    e1 = inputs[0]
    e2 = inputs[1]
    if self.feedforward:
        feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
        print('ff: ', feed_forward_product)
    e1 = K.flatten(inputs[0])
    e2 = K.flatten(inputs[1])
    e1 = K.tile(e1, k)
    e2 = K.tile(e2, k)
    e1 = K.reshape(e1, shape=(-1, input_size, k))
    e2 = K.reshape(e2, shape=(-1, input_size, k))
    x = e1 * (e2 * self.W)
    print('x:', x)
    y = self.collector(x, axis=-1, keepdims=True)
    z = K.squeeze(y, axis=-1)
    diag_tensor_products = z
    stacked = z # K.stack(diag_tensor_products)
    print('o1: ', stacked)
    #stacked = K.expand_dims(stacked, axis=0)
    print('o2: ', stacked)
    #print('o2: ', K.reshape(K.concatenate(diag_tensor_products, axis=1), (batch_size, k)))
    #print('o3: ', K.reshape(K.concatenate(diag_tensor_products, axis=1), (-1, k)) + feed_forward_product)
    #ff:  Tensor("neural_tensor_diag_layer_2/MatMul:0", shape=(?, 2048), dtype=float32)
    #o1:  Tensor("neural_tensor_diag_layer_2/stack:0", shape=(2048,), dtype=float32)
    if self.feedforward and self.bias:
        result = stacked + feed_forward_product + self.b
    elif self.feedforward:
        result = stacked + feed_forward_product  
    elif self.bias:
        result = stacked + self.b
    else:
        result = stacked
    if self.activation:
        result = self.activation(result)
        
    print('result: ', result)
    print("call() finished")
    return result

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)

  # not sure if valid when not base types
  def get_config(self):
    config = super(NeuralTensorDiagLayer, self).get_config()
    config.update({'output_dim': self.output_dim})
    config.update({'activation': self.activation})
    config.update({'collector': self.collector})
    config.update({'bias': self.bias})
    config.update({'feedforward': self.feedforward})
    return config

