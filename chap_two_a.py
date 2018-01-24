import tensorflow as tf
import numpy as np

def exampleVars():
	sess = tf.InteractiveSession()

	raw = [1.,2.,8.,-1.,0,5.5,6.,13]
	spikes = tf.Variable(False)
	spikes.initializer.run()

	for x in range(1,len(raw)):
		if raw[x] - raw[x-1] > 5:
			# sudden change
			tf.assign(spikes,True).eval()
		else:
			tf.assign(spikes,False).eval()
		print(spikes.eval())

	sess.close()

print("calculate exponential averaging")

def exampleExponentialAvg():

	# Constants
	alpha = tf.constant(0.05)
	# Initializing vars
	prev_avg = tf.Variable(0.)
	# Placeholder
	curr_val = tf.placeholder(tf.float32)


	# Equation:
	# Avg(t) =  A*X(t) + (1-A)*Avg(t-1)
	# operator:
	operator = alpha*curr_val + (1-alpha)*prev_avg


	raw_data = np.random.normal(10,1,100)


	# Creating graphs
	avg_summry = tf.summary.scalar("avg summary",prev_avg)
	value_summry = tf.summary.scalar("value summary",curr_val)
	merged_summry = tf.summary.merge_all()
	writer = tf.summary.FileWriter("./logs")

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		for i in range(len(raw_data)):
			summry_result,curr_avg = sess.run([merged_summry,operator],feed_dict={curr_val:raw_data[i]})
			tf.assign(prev_avg,curr_avg).eval()
			# print(prev_avg.eval())
			writer.add_summary(summry_result,i)

exampleExponentialAvg()
