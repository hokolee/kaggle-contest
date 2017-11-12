from utils import *
import math
import time


def cross_layer(x0, xl, scale=1.0, **kwargs):
  print("cross_layer")
  print(x0.get_shape())
  print(xl.get_shape())

  # assert x0.get_shape() == xl.get_shape()

  dim = x0.get_shape().ndims
  input_width = x0.get_shape()[dim - 1].value

  stddev = scale / math.sqrt(input_width)
  initializer = tf.truncated_normal_initializer(stddev=stddev)

  weights = tf.get_variable('weights', [input_width], tf.float32, initializer, **kwargs)
  biases = tf.get_variable('biases', [input_width], tf.float32, tf.zeros_initializer, **kwargs)

  x0xlt = tf.expand_dims(x0, -1) * tf.expand_dims(xl, -2)
  return tf.tensordot(x0xlt, weights, [[dim - 1], [0]]) + biases + xl


def weights_and_biases(layer1, layer2):
  # weights = tf.get_variable("weights", [layer1, layer2], initializer=tf.truncated_normal_initializer(stddev=0.1))
  weights = tf.get_variable("weights", [layer1, layer2], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(layer1)), seed=int(time.time())))
  # biases = tf.get_variable("biases", [layer2], initializer=tf.truncated_normal_initializer(stddev=0.1, seed=int(time.time())))
  biases = tf.get_variable("biases", [layer2], initializer=tf.zeros_initializer())
  return weights, biases


def full_connect_layer(inputs, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  return tf.matmul(inputs, weights) + biases


def sparse_matmul_layer(ids, values, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  inputs = tf.sparse_merge(ids, values, layer1)
  return tf.sparse_tensor_dense_matmul(inputs, weights) + biases


def sparse_embedding_layer(ids, values, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  # Use "mod" if skewed
  return tf.nn.embedding_lookup_sparse(weights, ids, values, partition_strategy="mod", combiner="mean") + biases


def inference_dnn(ids, values, dims):
  # Input layer
  with tf.variable_scope('input'):
    tmp0 = sparse_matmul_layer(ids, values, dims[0], dims[1])  # may tune
    # tmp1 = sparse_embedding_layer(ids, values, dims[0], dims[1])  # may tune
    out0 = tf.nn.relu(tmp0)
  # Hidden 1
  with tf.variable_scope('layer1'):
    tmp1 = full_connect_layer(out0, dims[1], dims[2])
    out1 = tf.nn.relu(tmp1)
  # Hidden 2
  with tf.variable_scope('layer2'):
    tmp2 = full_connect_layer(out1, dims[2], dims[3])
    out2 = tf.nn.relu(tmp2)

  # Output
  with tf.variable_scope('output'):
    logits = full_connect_layer(out2, dims[3], 1)
  return tf.nn.sigmoid(logits)


def inference_wide_and_deep(ids, values, dims):
  # Input layer
  with tf.variable_scope('input'):
    tmp0 = sparse_matmul_layer(ids, values, dims[0], dims[1])  # may tune
    # tmp1 = sparse_embedding_layer(ids, values, dims[0], dims[1])  # may tune
    out0 = tf.nn.relu(tmp0)
  # Hidden 1
  with tf.variable_scope('layer1'):
    tmp1 = full_connect_layer(out0, dims[1], dims[2])
    out1 = tf.nn.relu(tmp1)
  # Hidden 2
  with tf.variable_scope('layer2'):
    tmp2 = full_connect_layer(out1, dims[2], dims[3])
    out2 = tf.nn.relu(tmp2)

  # Output
  with tf.variable_scope('output_dense'):
    logits_dense = full_connect_layer(out2, dims[3], 1)
  with tf.variable_scope('output_sparse'):
    logits_sparse = sparse_matmul_layer(ids, values, dims[0], 1)  # may tune
  logits = logits_dense + logits_sparse
  return tf.nn.sigmoid(logits)


# deep cross
def inference_deep_cross_3layers(ids, values, dims):
  # Input layer
  with tf.variable_scope('input'):
    tmp0 = sparse_matmul_layer(ids, values, dims[0], dims[1])  # may tune
    # tmp0 = sparse_embedding_layer(ids, values, dims[0], dims[1])  # may tune
    out0 = tf.nn.relu(tmp0)

  # Hidden layer 1
  with tf.variable_scope('layer1'):
    tmp1 = full_connect_layer(out0, dims[1], dims[2])
    out1 = tf.nn.relu(tmp1)
  # Hidden 2
  with tf.variable_scope('layer2'):
    tmp2 = full_connect_layer(out1, dims[2], dims[3])
    out2 = tf.nn.relu(tmp2)

  print("tmp0")
  print(tmp0.get_shape())

  # Cross layer 1
  with tf.variable_scope('cross1'):
    cos1 = cross_layer(tmp0, tmp0)

  print("cos1")
  print(cos1.get_shape())
  print("tmp0")
  print(tmp0.get_shape())

  # Cross layer 2
  with tf.variable_scope('cross2'):
    cos2 = cross_layer(tmp0, cos1)

  # Output
  with tf.variable_scope('output'):
    output = tf.concat([out2, cos2], 1)
    print(output.get_shape())
    print(dims[1] + dims[3])
    logits = full_connect_layer(output, dims[1] + dims[3], 1)
  return tf.nn.sigmoid(logits)


def inference(ids, values, dims):
  # return inference_deep_cross_3layers(ids, values, dims)
  return inference_dnn(ids, values, dims)


def log_loss(labels, predictions):
  labels = tf.Print(labels, [labels], "labels: ", first_n=100, summarize=20)
  predictions = tf.Print(predictions, [predictions], "predictions: ", first_n=100, summarize=20)
  loss = tf.losses.log_loss(labels=labels, predictions=predictions)
  loss = tf.Print(loss, [loss], "loss: ", first_n=100, summarize=200)
  return loss

