#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import xlrd
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state

# generating dataset for training the model, 50 training samples included.
n = 50											# number of samples
XX = np.arange(n)								# coordinates of X(i)
rs = check_random_state(0)						# type of distribution?
YY = rs.randint(-20, 20, size=(n,)) + 2.0 * XX	# generate values of Y(i)
data = np.stack([XX,YY], axis=1)
												# set up the actual
												# training data.

# defining the flags
log_dir = tf.app.flags.DEFINE_string('log_dir',
	os.path.dirname(os.path.abspath(__file__)) + '/log',
	'Directory of event files.')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Description...')
FLAGS = tf.app.flags.FLAGS

# creating the variable needed
W = tf.Variable(0.0, name = "weight")	# A.K.A. slope/die Steigung
b = tf.Variable(0.0, name = "bias")		# y(pridiction) = w*x + b

# the relvant functions in training
def inputs():
	"""
	first time using massive comment.
	"""
	X = tf.placeholder(tf.float32, name = "X")
	Y = tf.placeholder(tf.float32, name = "Y")

	return X,Y

def inference(X):
	return W * X + b

def loss(X, Y):
	Y_predicted = inference(X)
	return tf.squared_difference(Y, Y_predicted)

def train(loss):
	learning_rate = 0.0001
	return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# the main body
with tf.Session() as sess:
	# in TensorFlow the functions would only be executed
	# when called with sess.run(function()).
	# Otherwise the code written inside this part is still
	# creating names/variables...

	# initialization
	sess.run(tf.global_variables_initializer())

	# the flow-chart of training process
	X, Y = inputs()

	train_loss = loss(X, Y)
	train_op = train(train_loss)
	#writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
	for epoch_num in range(FLAGS.num_epochs):
		
		for x, y in data:
			# to be clarified: what does the name of "loss_value_," mean?
			#train_op = train(train_loss)
			loss_value, _ = sess.run([train_loss,train_op], 
									 feed_dict={X: x, Y: y})
			#print('epoch %d, loss=%f' %(epoch_num+1, loss_value))
			#print(epoch_num+1, loss_value)
			print(x,y,loss_value)
			w, b = sess.run([W, b])
		wcoeff, bias = sess.run([W, b])
	print(w, b)
	print(wcoeff, bias)

#writer.close()
#sess.close()
###############################
#### Evaluate and plot ########
###############################
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * w + b
Prediction_values_final = data[:,0] * wcoeff + bias

# uncomment if plotting is desired!
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')
plt.plot(Input_values, Prediction_values_final, label='Final Prediction')

# # Saving the result.
plt.legend()
plt.savefig('plot.png')
plt.close()