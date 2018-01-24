import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

def LinearReg():
	x_train = np.linspace(-1,1,101)
	# y = x + noise
	# => y = 2x + noise*0.33
	y_train = 2*x_train + np.random.randn(*x_train.shape) * 0.33

	# hyper params
	epochs = 100
	learning_rate = 0.08
	# ,0.08,0.1,0.12,0.14,0.16
	listOfLearningRates = [0.04]

	# place holder nad variables
	X = tf.placeholder(np.float32) * x_train.shape
	Y = tf.placeholder(np.float32) * y_train.shape

	W0 = tf.Variable([.0]*len(x_train.shape),name='parameters')
	W1 = tf.Variable([.0]*len(x_train.shape),name='parameters')

	# W1 = tf.Variable(0.0)
	# W0 = tf.Variable(0.0)

	def model(someX,W1,W0):
		return tf.add(W0,tf.multiply(someX,W1))

	modelOp = model(X,W1,W0)

	cost = tf.square(Y-modelOp)
	op = []
	for rate in listOfLearningRates:
		op.append(tf.train.GradientDescentOptimizer(rate).minimize(cost))
	init = tf.global_variables_initializer()

	learnedW1 = []
	learnedW0 = []	
	with tf.Session() as sess:
		sess.run(init)
		# tf.trainable_variables()
		# for epoch in range(epochs):
		for operator in op:
			for (x,y) in zip(x_train,y_train):
				sess.run(operator,feed_dict={X:x,Y:y})
			learnedW0.append(sess.run(W0))
			learnedW1.append(sess.run(W1))

			# learnedW0.append(tf.cast(sess.run(W0), tf.float32))
			# learnedW1.append(tf.cast(sess.run(W1), tf.float32))

	plt.scatter(x_train,y_train)

	for w0,w1 in zip(learnedW0,learnedW1):
		m = w0+x_train*w1
		plt.plot(x_train, m)

	plt.show()


# LinearReg()

def PolynomialReg():
	x_train = np.linspace(-1,1,101)

	# y = w0 X^0 + w1 X^1 + w2 X^2+ .. + wn X^n
	# degree = 6


	degrees = [1.,2.,3.,4.,5.,6.]
	y_train = 0
	for i in range(0,len(degrees)):
		y_train += degrees[i]*np.power(x_train,i)
	y_train += np.random.randn(*x_train.shape) * 1.5
	# plt.scatter(x_train,y_train)
	# plt.show()

	# defining Model:
	def model(X,W):
		y = []
		for i in range(1,len(degrees)):
			y.append(tf.multiply(W[i] , tf.pow(X,i)))

		return tf.add_n(y)


	# # hyper params
	epochs = 40
	learning_rate = 0.01

	# # place holder nad variables
	X = tf.placeholder("float") * x_train.shape
	Y = tf.placeholder("float") * y_train.shape

	W = tf.Variable([0.0]*len(degrees),name='parameters')

	# def model(someX,someW):
	# 	return tf.multiply(someX,someW)

	modelOp = model(X,W)

	cost = tf.square(Y-modelOp)

	op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		tf.trainable_variables()
		for epoch in range(epochs):
			for (x,y) in zip(x_train,y_train):
				sess.run(op,feed_dict={X:x,Y:y})
			w_wal = sess.run(W)

	plt.scatter(x_train,y_train)
	y_learned = 0
	for i in range(0,len(degrees)):
		y_learned += np.power(x_train,i)*w_wal[i]
	plt.plot(x_train, y_learned)
	plt.show()

PolynomialReg()