# @author: BOYU ZHANG
# GitHub: CristianoBY
# All Rights Reserved

import tensorflow as tf 
import numpy as np 

# the implementation of the XNOR gate in 2-layer neural network
# using sigmoid for the all layers' activation and square error as a cost function
# using AdamOptimizer for the neural network
# XNOR = AND || (NOT x1)AND(NOT x2)

# traning parameters
learning_rate = 0.03
training_epochs = 5000


# training data: only [1, 1] or [0, 0] yeild true XNOR ([1])
train_X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_Y = np.array([[1], [0], [0], [1]])

# tf Graph input of any size
x = tf.placeholder(tf.float32, [None, np.size(train_X, 1)])
y = tf.placeholder(tf.float32, [None, np.size(train_Y, 1)])


# randomly initialized weight for each layer
weights = {

	# first WEIGHT matrix size: number of features in an example * 2
	# weights for AND && (NOT x1)AND(NOT x2), respectively
    'W1': tf.Variable(tf.random_normal([np.size(train_X, 1), 2])),

    # second hidden layer size: 2 * 1
    # take in AND || (NOT x1)AND(NOT x2) and output XNOR by OR
    'W2': tf.Variable(tf.random_normal([2, 1]))

}

# randomly initialized bias for each layer
biases = {

    'b1': tf.Variable(tf.random_normal([2])), # bias is 1 * 2 for AND && (NOT x1)AND(NOT x2), respectively

    'b2': tf.Variable(tf.random_normal([1])) # bias is -10 for OR

}

# generate prediction by FP in NN
def forward_propagation(x):

	# one example: 1 * number of features
	l1 = x

	with tf.variable_scope("layer1"):
		# second layer: 1 * 2 (AND && (NOT x1)AND(NOT x2))
		l2 = tf.add(tf.matmul(l1, weights['W1']), biases['b1'])

		# activation by sigmoid
		a2 = tf.nn.sigmoid(l2)

	with tf.variable_scope("layer2"):
		# 1 * 1 (take in AND && (NOT x1)AND(NOT x2) then output XNOR)
		l3 = tf.add(tf.matmul(a2, weights['W2']), biases['b2'])

		# activation by sigmoid
		a3 = tf.nn.sigmoid(l3)

	# return the output layer
	return a3

# cost function of sigmoid activation with square error
cost = tf.reduce_mean(tf.square(forward_propagation(x) - y)) / np.size(train_X, 0)

# AdamOptimizer for the neural network
training_process = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# run the learning algorithm
with tf.Session() as sess:
    sess.run(init)

    # initial weight & bias
    print("\nweights before traning: \n", "w1: ", weights['W1'].eval(), "\n", "w2: ", weights['W2'].eval(), "\n") 
    print("bias before traning: \n", "b1: ", biases['b1'].eval(), "\n", "b2: ", biases['b2'].eval(), "\n") 

    # train
    for i in range(training_epochs):

    	sess.run(training_process, feed_dict = {x: train_X, y: train_Y})

    print("Optimization Finished (sigmoid activation)\n")

    # final weight & bias  
    print("weights after traning: \n", "w1: ", weights['W1'].eval(), "\n", "w2: ", weights['W2'].eval()) 
    print("\nbias after traning: \n", "b1: ", biases['b1'].eval(), "\n", "b2: ", biases['b2'].eval(), "\n") 

    # test the result
    final_cost = sess.run(cost, feed_dict = {x: train_X, y: train_Y})
    print("final remaining error: \n", final_cost)

    final_pred = sess.run(forward_propagation(x), feed_dict = {x: train_X})
    print("\nfinal prediction: \n", final_pred)
