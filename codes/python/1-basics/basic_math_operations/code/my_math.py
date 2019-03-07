#!/usr/bin/env python3

# include the libs
from __future__ import print_function
import tensorflow as tf
import os

# define the directory of event files

# flags is a structure of storing data
dir_log = tf.app.flags.DEFINE_string('dir_log',
os.path.dirname(os.path.abspath(__file__)) + '/logs',
'Place of storing event files.')

# store the data in an object
FLAGS = tf.app.flags.FLAGS

# do the math
welcome = tf.constant('using basic calculations in TensorFlow.')
a = tf.constant(5.0, name = "a")
b = tf.constant(10.0, name = "b")

x = tf.add(a, b, name = "add")
y = tf.div(a, b, name = "divide")

with tf.Session() as sess:
	writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.dir_log), sess.graph)
	print("Output: ", sess.run([welcome, a, b, x, y]))

# closing the writer & Session
writer.close()
sess.close()