import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test1():
	x_train = np.linspace(-1,1,101)

	# y = x + noise
	# => y = 2x + noise*0.33
	y_train = 2*x_train + np.random.randn(*x_train.shape) * 0.33

	plt.scatter(x_train,y_train)

	plt.plot(x_train,y_train)

	plt.show()

# test1()