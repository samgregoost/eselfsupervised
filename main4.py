import tensorflow as tf
import random
#import tf.keras.layers.Dense
import numpy as np
import math as ma

def fun(m,x):
	return m*x

def fun2(m,x):
        return m*ma.pow(x,2)

def fun3(m,x):
        return m*ma.pow(x,3)


for x in range(10):
	k = random.uniform(0, 1)
	y = fun(4,k)
	print(str(k) + "," + str(y))


learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)


	
X = tf.placeholder("float", [None, 1])
Y = tf.placeholder("float", [None, 3])
Z = tf.placeholder("float", [None, 1])


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def set_batch(batch_size, m):
	x_array = np.array([])
	y_array = np.array([])
	for x in range(batch_size):
		y_ind =  np.array([])
		k = random.uniform(0, 1)
		y1 = fun(m,k)
		y2 = fun2(m,k)
		y3 = fun3(m,k)
		y = np.array((y1,y2,y3))
		x_array = np.append(x_array, k)
		y_array = np.append(y_array, y)

	x_array = np.reshape(x_array,(-1,1))
	y_array = np.reshape(y_array,(-1,3))
	
	return x_array, y_array



# Create model
def pred(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.compat.v1.layers.dense(x,256))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.compat.v1.layers.dense(layer_1,256))
    # Output fully connected layer with a neuron for each class
    out_layer =  tf.compat.v1.layers.dense(layer_2,3)
    return out_layer

def dec(h,z):
    # Hidden fully connected layer with 256 neurons
    print(h)
    print(z)
    h_ = tf.concat([h,z],axis = 1)	
    layer_1 = tf.nn.relu(tf.compat.v1.layers.dense(h_,256))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.compat.v1.layers.dense(layer_1,256))
    # Output fully connected layer with a neuron for each class
    out_layer =  tf.compat.v1.layers.dense(layer_2,3)
    return out_layer



h = pred(X)
y_ = dec(h,Z)

loss = tf.reduce_mean(tf.squared_difference(y_, Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
# Initializing the variables

grads = tf.gradients(ys = loss, xs = Z)
init = tf.global_variables_initializer()

#grads = tf.gradients(loss, Z)
#grads, _ = tf.clip_by_global_norm(grads, 50) # gradient clipping
#grads_and_vars = list(zip(grads, Z))
#train_op_z = optimizer.apply_gradients(grads_and_vars)


#grads = tf.gradients(ys = loss, xs = Z)


with tf.Session() as sess:
	sess.run(init)

	for epoch in range(10000):
		batch_x, batch_y = set_batch(batch_size,4)

#		z = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
		z = np.reshape(np.random.uniform(low=0.0, high=1.0, size=(100,1)),(-1,1))
	#	_, c = sess.run([train_op, loss], feed_dict={X: batch_x,
                    #                                        Y: batch_y, Z:z})
		v = 0
#		for p in range(1000):
#			g = sess.run([grads],feed_dict={X: batch_x,Y: batch_y,Z:z})
#			print(g[0][0])
#			v_prev = np.copy(v)
#			v = 0.01*v - 0.001*g[0][0]
#			z += 0.01 * v_prev + (1+0.01)*v
#			z = np.clip(z, -1, 1)			


		_, c = sess.run([train_op, loss], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z})
		batch_x_, batch_y_ = set_batch(batch_size,-4)
		#z_ = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
	#	z = np.reshape(np.array(random.uniform(0, 1)),(-1,1))
		z = np.reshape(np.random.uniform(low=0.0, high=1.0, size=(100,1)),(-1,1))
		#_, c = sess.run([train_op, loss], feed_dict={X: batch_x_,
                #                                            Y: batch_y_, Z:z_})
		v = 0
#		for p in range(1000):
#			g = sess.run([grads],feed_dict={X: batch_x,Y: batch_y,Z:z})
#			v_prev = np.copy(v)
#			v = 0.01*v - 0.001*g[0][0]
#			z += 0.01 * v_prev + (1+0.01)*v
#			z = np.clip(z, -1, 1)

		_, c = sess.run([train_op, loss], feed_dict={X: batch_x_,
                                                            Y: batch_y_, Z:z})
		print(c)



	


	for x in range(10):
		k = random.uniform(0, 1)
		k_ = np.reshape(k,(-1,1))
	#	z = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
		z = np.reshape(np.array(random.uniform(0, 1)),(-1,1))
		y = sess.run([y_],  feed_dict={X:k_,Z:z})
		print(str(k) + "," + str(y))













