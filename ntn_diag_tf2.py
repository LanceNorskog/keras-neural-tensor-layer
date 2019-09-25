
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
import scipy.stats as stats

class NeuralTensorDiagLayer(Layer):
  def __init__(self, output_dim, activation=K.tanh, collector=K.mean, **kwargs):
    self.output_dim = output_dim #k
    self.activation = activation
    self.collector = collector
    super(NeuralTensorDiagLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    assert input_shape[0][-1].value == input_shape[1][-1].value
    mean = 0.0
    std = 1.0
    # W : k*d
    k = self.output_dim
    print('Input shapes: ' + str(input_shape))
    print('Input shape: ' + str(input_shape[0]))
    print('Input shape[-1]: ' + str(input_shape[0]))
    d = input_shape[0][-1].value

    print('d: ' + str(d))
    w_init = tf.initializers.truncated_normal(mean=mean, stddev=2*std)
    self.W = self.add_weight(shape=(k, d), 
                             initializer=w_init,
                             trainable=True,
                             name='W')
    self.V = self.add_weight(shape=(2*d, k), 
                             initializer=w_init,
                             trainable=True,
                             name='V')
    # stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d))
    self.b = K.zeros((k), name='b')
    #self.trainable_weights = [self.W, self.V, self.b]
    print("build() finished")


  def call(self, inputs, mask=None):
    print("call(): ", inputs)
    if type(inputs) is not list or len(inputs) != 2:
      raise Exception('NTNDiagLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    print('batch_size: ', batch_size)
    k = self.output_dim
    print('inputs: ', [e1,e2])
    feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
    print('ff: ', feed_forward_product)
    print('d1: ', e1 * self.W[0])
    print('d2: ', e2 * (e1 * self.W[0]))
    #print('d3: ', e2 * (e1 * self.W[0]) + self.b)
    diag_tensor_products = [] 
    for i in range(k):
      diag_tensor_products.append(self.collector(e2 * (e1 * self.W[i])))
    print('diag.shape: ', K.shape(diag_tensor_products[0]))
    print('o1: ', K.stack(diag_tensor_products))
    #print('o2: ', K.reshape(K.concatenate(diag_tensor_products, axis=1), (batch_size, k)))
    #print('o3: ', K.reshape(K.concatenate(diag_tensor_products, axis=1), (-1, k)) + feed_forward_product)
    stacked = K.stack(diag_tensor_products) + feed_forward_product + self.b
    result = self.activation(stacked)
    print('result: ', result)
    print("call() finished")
    return result

  def compute_output_shape(self, input_shape):
    print ("compute_output: ")
    batch_size = input_shape[0][0]
    print ("compute_output: ", str((batch_size, self.output_dim)))

    return (batch_size, self.output_dim)

  def get_config(self):
    config = super(NeuralTensorDiagLayer, self).get_config()
    config.update({'output_dim': self.output_dim})
    return config
