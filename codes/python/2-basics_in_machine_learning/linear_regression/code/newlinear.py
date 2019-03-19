#!/usr/bin/env python3

# importing the packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
#import os		# uncomment when log file is required
#import xlrd	# uncomment when reading data set from .xls file

# creating sample data set
n = 50
rs = check_random_state(0)
x = np.arange(n)					# rs.rand(n) creates a vector consists
									# of n numbers randomly distrubute on [0,1]
y = rs.randint(-20, 20, (n,)) + 2.0 * x

# defining variables needed
W = tf.Variable(0.0, name = 'weight')
b = tf.Variable(0.0, name = 'bias')
rounds_total = 100

# defining functions
def inputs():
	X = tf.placeholder(tf.float32, name = "x")
	Y = tf.placeholder(tf.float32, name = "y")
	return X, Y

def prediction(X):
	return W * X + b

def loss(Y, Y_predicted):
	return (tf.reduce_sum(tf.squared_difference(Y, Y_predicted)))/n

def train(loss):
	learning_rate = 0.0001
	return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	#	tf.train.GradientDescentOptimizer(lr).minimize(loss)


# defining functions
with tf.Session() as sess:
	# initializing the variables
	sess.run(tf.global_variables_initializer())

	# set pseu-variables
	X, Y = inputs()
	# difining operations
	get_loss_op = loss(Y, prediction(X))
	get_train_op = train(get_loss_op)
	# defining main body of algorithm
	for round in np.arange(rounds_total):
		current_loss, current_trained = sess.run([get_loss_op, get_train_op], feed_dict = {X: x, Y: y})
		print(sess.run([W, b]), current_loss)
	wf, bf = sess.run([W, b])
print(wf, bf)



sess.close()

# plotting the image to .png
plt.plot(x, y, 'ro', label = "Saple value")
plt.plot(x, wf * x + bf, label = "Predicted value")
plt.legend()
plt.savefig("result.png")
plt.close()