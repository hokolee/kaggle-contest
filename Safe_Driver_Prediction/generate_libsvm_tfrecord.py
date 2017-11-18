#!/usr/bin/env python

import tensorflow as tf
import sys
import os

def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)

  for line in open(input_filename, "r"):
    data = line.split(" ")
    label = float(data[0])
    _, qid = data[1].split(":")
    ids = []
    values = []
    for fea in data[2:]:
      id, value = fea.split(":")
      ids.append(int(id))
      values.append(float(value))

    # Write each example one by one
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        "instance_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(qid)])),
        "feature_id": tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=values))
    }))

    writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_filename,output_filename))

def main():
  for arg in sys.argv[1:]:
    print("input {}".format(arg))
    for filename in os.listdir(arg):
      print(filename)
      if filename.startswith("") and filename.endswith(".libsvm"):
        generate_tfrecords(os.path.join(arg, filename), os.path.join(arg, filename + ".tfrecords"))

if __name__ == "__main__":
  main()