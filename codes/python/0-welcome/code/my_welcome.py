#!/usr/bin/env python3

from __future__ import print_function
import tensorflow as tf
import os
# os = operating system

tf.app.flags.DEFINE_string('log_dir', 
os.path.dirname(os.path.abspath(__file__)) + '/logs',
'Directory where event logs are written to.')

# storing all the flags in a structure FLAGS
# the flags could be called with FLAGS.flag_name
FLAGS = tf.app.flags.FLAGS

if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
	raise ValueError('An absolute path must be assigned to --log_dir')

welcome = tf.constant('Welcome to the world of TensorFlow!')

# Run the session
with tf.Session() as sess:
	writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
	print('Output: ', sess.run(welcome))

writer.close()
sess.close()