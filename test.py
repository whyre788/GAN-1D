import tensorflow as tf
import numpy as np
import os
import scipy.io as sio

flags = tf.app.flags
flags.DEFINE_string("dataset", "./Data/data4train.mat", "The path to dataset")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Directory name to save the checkpoints")
flags.DEFINE_string("output_dir", "./output/", "Directory name to save the test output")
FLAGS = flags.FLAGS

matfn=FLAGS.dataset
f = sio.loadmat(matfn)
fake_data = f['x0']
fake_data = np.array(fake_data)
if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
try:
    restore_dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
except Exception:
    raise Exception("[!] Train a model first, then run test mode")
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
with sess.graph.as_default():
    saver = tf.train.import_meta_graph(restore_dir+'.meta')
    saver.restore(sess, restore_dir)
    output = tf.get_collection('GAN_1D')[0]
    fake_X = graph.get_operation_by_name('fake_input').outputs[0]
    output = sess.run(output,feed_dict={fake_X:fake_data})
    output = np.array(output)
np.savetxt(FLAGS.output_dir+"test_data.csv", output, fmt="%f", delimiter=",")
