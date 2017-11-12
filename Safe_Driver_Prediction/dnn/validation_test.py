from network import *
from sklearn.metrics import roc_auc_score
import numpy as np

def output_validation(cur_epoch, cur_loss, cur_instance_id, cur_prediction, cur_label):
  ans = []
  for it in range(0, len(cur_instance_id)):
    assert (len(cur_instance_id[it]) == len(cur_prediction[it]))
    assert (len(cur_instance_id[it]) == len(cur_label[it]))
    for jt in range(0, len(cur_instance_id[it])):
      line = "%d,%.15f,%.2f\n" % (cur_instance_id[it][jt][0], cur_prediction[it][jt][0], cur_label[it][jt][0])
      ans.append(line)
  output_file = "%s/validation_%f_%09d.csv" % (FLAGS.result_dir, cur_loss, cur_epoch)
  with open(output_file, 'wb') as csv_file:
    csv_file.write("instance_id,proba,label\n")
    for line in sorted(ans):
      csv_file.write(line)
  logger.info('output result to %s done', output_file)
  csv_file.close()


def calc_validation(sess, batch, epoch):
  label = tf.placeholder(tf.float32)
  feature_id = tf.sparse_placeholder(tf.int64)
  value = tf.sparse_placeholder(tf.float32)

  tf.get_variable_scope().reuse_variables()
  prediction = inference(feature_id, value, layers)
  mean_loss = log_loss(label, prediction)

  ans_length = 0
  ans_sum_loss = 0
  ans_instance_id = []
  ans_prediction = []
  ans_label = []
  for x in range(len(batch)):
    cur_label = batch[x]['label']
    cur_instance_id = batch[x]['instance_id']
    cur_mean_loss, cur_prediction = sess.run([mean_loss, prediction], feed_dict={
      label: batch[x]['label'],
      feature_id: batch[x]['feature_id'],
      value: batch[x]['value']
    })
    cur_length = len(cur_label)
    ans_length += cur_length
    ans_sum_loss += cur_mean_loss * cur_length
    ans_instance_id.append(cur_instance_id)
    ans_prediction.append(cur_prediction)
    ans_label.append(cur_label)
    # print(len(cur_instance_id), len(cur_prediction), len(cur_label))
  logger.info('%d, %d, %d, %d', len(batch), len(ans_instance_id), len(ans_prediction), len(ans_label))
  ans_mean_loss = ans_sum_loss / ans_length
  logger.info('%f, %d, %f', ans_sum_loss, ans_length, ans_mean_loss)
  output_validation(epoch, ans_mean_loss, ans_instance_id, ans_prediction, ans_label)

  np_label = np.array([element for sub in ans_label for element in sub])
  np_prediction = np.array([element for sub in ans_prediction for element in sub])
  roc_auc = roc_auc_score(np_label, np_prediction)
  logger.info('validation epoch: %d, batch %d, roc_auc: %f, gini: %f', epoch, len(batch), roc_auc, 2*roc_auc -1)

  return ans_mean_loss


def output_test(cur_epoch, cur_loss, cur_instance_id, cur_prediction):
  ans = []
  for it in range(0, len(cur_instance_id)):
    assert (len(cur_instance_id[it]) == len(cur_prediction[it]))
    for jt in range(0, len(cur_instance_id[it])):
      line = "%d,%.15f\n" % (cur_instance_id[it][jt][0], cur_prediction[it][jt][0])
      ans.append(line)
  output_file = "%s/test_%f_%09d.csv" % (FLAGS.result_dir, cur_loss, cur_epoch)
  with open(output_file, 'wb') as csv_file:
    csv_file.write("id,target\n")
    for line in sorted(ans):
      csv_file.write(line)
  logger.info('output result to %s done', output_file)
  csv_file.close()


def calc_test(sess, batch, validation_loss, epoch):
  label = tf.placeholder(tf.float32)
  feature_id = tf.sparse_placeholder(tf.int64)
  value = tf.sparse_placeholder(tf.float32)

  tf.get_variable_scope().reuse_variables()
  prediction = inference(feature_id, value, layers)

  ans_instance_id = []
  ans_prediction = []
  for x in range(len(batch)):
    cur_instance_id = batch[x]['instance_id']
    cur_prediction = sess.run(prediction, feed_dict={
      label: batch[x]['label'],
      feature_id: batch[x]['feature_id'],
      value: batch[x]['value']
    })
    ans_instance_id.append(cur_instance_id)
    ans_prediction.append(cur_prediction)
    # print(len(cur_instance_id), len(cur_prediction))
  logger.info('%d, %d, %d', len(batch), len(ans_instance_id), len(ans_prediction))
  output_test(epoch, validation_loss, ans_instance_id, ans_prediction)
