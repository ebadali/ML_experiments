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

	# place holder nad variables
	X = tf.placeholder(np.float32) * x_train.shape
	Y = tf.placeholder(np.float32) * y_train.shape

	W = tf.Variable(0.0)

	def model(someX,someW):
		return tf.multiply(someX,someW)

	modelOp = model(X,W)

	cost = tf.square(Y-modelOp)

	op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.initialize_all_variables()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			for (x,y) in zip(x_train,y_train):
				sess.run(op,feed_dict={X:x,Y:y})
			w_wal = sess.run(W)

	plt.scatter(x_train,y_train)
	y_learned = x_train*w_wal
	plt.plot(x_train, y_learned)
	plt.show()

LinearReg()

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


def normalize(ary):
	# Find min and max,
	print(np.amax(ary))
	return ary

# PolynomialReg()
def parsing(fileName):
	price = []
	numrooms = []

	with open(fileName,'rb') as csvFile:
		# data = np.genfromtxt(csvFile,delimiter=",", skip_header=2)
		# data = list(csv.reader(csvFile))
		# print(data[0:1,1])
		reader = csv.reader(csvFile)
		# Skipping first line
		reader.next()
		for row in reader:
			price.append(row[5])
			numrooms.append(row[0])
	# return data[0:2,2] , data[0:2,5]
	
	return normalize(np.array(numrooms,dtype=np.float)),normalize(np.array(price,dtype=np.float))
# print(parsing('USA_Housing.csv'))
def housing():
	y_train,x_train = parsing('USA_Housing.csv')

	# hyper params
	epochs = 10
	learning_rate = 0.003

	# place holder nad variables
	X = tf.placeholder(np.float32) * x_train.shape
	Y = tf.placeholder(np.float32) * y_train.shape

	W = tf.Variable(0.0)

	def model(someX,someW):
		return tf.multiply(someX,someW)

	modelOp = model(X,W)

	cost = tf.square(modelOp-Y)

	op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			for (x,y) in zip(x_train,y_train):
				sess.run(op,feed_dict={X:x,Y:y})
			w_wal = sess.run(W)

	plt.scatter(x_train,y_train)
	y_learned = x_train*w_wal
	plt.plot(x_train, y_learned)
	plt.show()

housing()

# y_train,x_train = parsing('USA_Housing.csv')

# plt.scatter(x_train,y_train)
# y_learned = x_train*w_wal
# plt.plot(x_train, y_train)
# plt.show()


