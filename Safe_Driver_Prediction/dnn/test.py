import tensorflow as tf
import numpy as np
import math


def cross_layer2(x, scale=1.0, **kwargs):
  dim = x.get_shape().ndims
  input_width = x.get_shape()[dim - 1].value

  stddev = scale / math.sqrt(input_width)
  initializer = tf.truncated_normal_initializer(stddev=stddev)

  w = tf.get_variable('weights', x.get_shape(), tf.float32, tf.ones_initializer, **kwargs)
  b = tf.get_variable('biases', x.get_shape(), tf.float32, tf.ones_initializer, **kwargs)

  xt = tf.transpose(x)
  wt = tf.transpose(w)
  xxt = tf.matmul(xt, x)
  # return tf.matmul(w, xxt) + x + b
  return tf.matmul(w, xxt)


def cross_layer(x, scale=1.0, **kwargs):
  dim = x.get_shape().ndims
  input_width = x.get_shape()[dim - 1].value

  print(input_width)

  stddev = scale / math.sqrt(input_width)
  initializer = tf.truncated_normal_initializer(stddev=stddev)

  # w = tf.get_variable('weights', [input_width], tf.float32, initializer, **kwargs)
  # b = tf.get_variable('biases', [input_width], tf.float32, tf.zeros_initializer, **kwargs)
  w = tf.get_variable('weights', [input_width], tf.float32, tf.ones_initializer, **kwargs)
  b = tf.get_variable('biases', [input_width], tf.float32, tf.ones_initializer, **kwargs)

  xxt = tf.expand_dims(x, -1) * tf.expand_dims(x, -2)
  # return tf.tensordot(xxt, w, [[dim - 1], [0]]) + x + b
  return tf.tensordot(xxt, w, [[dim - 1], [0]])


def cross_layer_bak(x0, xl, scale=1.0, **kwargs):
  assert x0.get_shape() == xl.get_shape()

  dim = x0.get_shape().ndims
  input_width = x0.get_shape()[dim - 1].value

  print(x0.get_shape())

  stddev = scale / math.sqrt(input_width)
  initializer = tf.truncated_normal_initializer(stddev=stddev)

  weights = tf.get_variable('weights', [input_width], tf.float32, initializer, **kwargs)
  biases = tf.get_variable('biases', [input_width], tf.float32, tf.zeros_initializer, **kwargs)

  x0xlt = tf.expand_dims(x0, -1) * tf.expand_dims(xl, -2)
  return tf.tensordot(x0xlt, weights, [[dim - 1], [0]]) + xl + biases


def cross_layer_right(x0, xl, scale=1.0, **kwargs):
  assert x0.get_shape() == xl.get_shape()

  dim = x0.get_shape().ndims
  input_width = x0.get_shape()[dim - 1].value

  print(x0.get_shape())

  stddev = scale / math.sqrt(input_width)
  initializer = tf.truncated_normal_initializer(stddev=stddev)

  weights = tf.get_variable('weights', [input_width], tf.float32, initializer, **kwargs)
  biases = tf.get_variable('biases', [input_width], tf.float32, tf.zeros_initializer, **kwargs)

  x0xlt = tf.expand_dims(x0, -1) * tf.expand_dims(xl, -2)
  return tf.tensordot(x0xlt, weights, [[dim - 1], [0]]) + xl + biases


tf.reset_default_graph()

# x = np.random.normal(size=(2,3,4))
x = np.random.normal(size=(2,4))
X = tf.placeholder_with_default(x.astype(np.float32), x.shape)

x0 = np.random.normal(size=(2,4))
X0 = tf.placeholder_with_default(x0.astype(np.float32), x0.shape)

xl = np.random.normal(size=(2,4))
XL = tf.placeholder_with_default(xl.astype(np.float32), xl.shape)

with tf.variable_scope('cross'):
  Y = cross_layer(X)

with tf.variable_scope('cross2'):
  Y2 = cross_layer2(X)

with tf.variable_scope('cross_right'):
  YRight = cross_layer_right(X0, XL)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  y = sess.run(Y)
  y2 = sess.run(Y2)
  y_right = sess.run(YRight)

  print(x.shape)
  print(x)

  print(x0.shape)
  print(x0)

  print(xl.shape)
  print(xl)

  print(y.shape)
  print(y)

  print(y2.shape)
  print(y2)

  print(y_right.shape)
  print(y_right)

  assert x.shape == y.shape
