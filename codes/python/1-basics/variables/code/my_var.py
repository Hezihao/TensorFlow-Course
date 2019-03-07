#!/usr/bin/env python3

# import the packages
import tensorflow as tf
from tensorflow.python.framework import ops		# a class?
import os

# define the directory of logs
log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs'

# defining some variables
weights = tf.Variable(tf.random_normal([2, 3], stddev = 0.1), 
name = "weights")
# for those 2 variables below, just the structures(but no data)
# is decleared here. Defined as defaults
biases = tf.Variable(tf.zeros([3]), name = "biases")
custome_variable = tf.Variable(tf.zeros([3]), name = "custome")	# tf.zeros([m,n])

# store all variables in a list
# GraphKeys : the index of nodes in the whole graph
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
custome_variable_list = [weights, custome_variable]

# define those initializers
initializer_all_op = tf.variables_initializer(var_list = all_variables_list)
initializer_custome_op = tf.variables_initializer(var_list = custome_variable_list)

# perform the initialization in a session
with tf.Session() as sess:
	writer = tf.summary.FileWriter(os.path.expanduser(log_dir), sess.graph)
	# sees.run() just incluenced the result of running the program
	# the graph in tensorboard will only be incluenced by the includings of
	# tensor-space
	sess.run(initializer_custome_op)
	sess.run(initializer_all_op)

writer.close()
sess.close()