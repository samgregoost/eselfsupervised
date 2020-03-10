import tensorflow as tf
import random
#import tf.keras.layers.Dense
import numpy as np
import math as ma
import matplotlib.pyplot as plt

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


learning_rate = 0.0001
training_epochs = 15
batch_size = 3
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)


	
X = tf.placeholder("float", [None, 1])
Y = tf.placeholder("float", [None, 3])
Z = tf.placeholder("float", [None, 1])
Z_old = tf.placeholder("float", [None, 1])
X_old = tf.placeholder("float", [None, 1])
Z_new = tf.placeholder("float", [None, 1])
Y_new = tf.placeholder("float", [None, 3])
Y_old = tf.placeholder("float", [None, 3])


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



def set_batch(batch_size):
	x_array = np.array([])
	y_array = np.array([])
	for x in range(batch_size):
		randnum = random.randint(1,101)
		if randnum > 50:
			maxx = 4
		else:
			maxx = -4
		m = maxx		
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
    out_layer =  tf.nn.sigmoid(tf.compat.v1.layers.dense(layer_2,3))
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

def z_pred(z,x):
    # Hidden fully connected layer with 256 neurons
    print(h)
    print(z)
    h_ = tf.concat([x,z],axis = 1)
    layer_1 = tf.nn.relu(tf.compat.v1.layers.dense(h_,256))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.compat.v1.layers.dense(layer_1,256))
    # Output fully connected layer with a neuron for each class
    out_layer =  tf.compat.v1.layers.dense(layer_2,1)
    return out_layer

def mag_pred(z,x):
    # Hidden fully connected layer with 256 neurons
    print(h)
    print(z)
    h_ = tf.concat([x,z],axis = 1)
    layer_1 = tf.nn.relu(tf.compat.v1.layers.dense(h_,256))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.compat.v1.layers.dense(layer_1,256))
# Output fully connected layer with a neuron for each class
    out_layer =  tf.nn.sigmoid(tf.compat.v1.layers.dense(layer_2,1))
    return out_layer


h = pred(X)
#z_ = z_enc(Z)
y_ = dec(h,Z)


z_new_pred = z_pred(Z_old,X_old)
magnitude_pred =  mag_pred(Z_old,X_old)
k_energy =  tf.clip_by_value(tf.math.square(magnitude_pred ),0.0,10.0)


z_loss = tf.reduce_mean(tf.squared_difference(z_new_pred , Z_new))



loss = tf.reduce_mean(tf.squared_difference(y_, Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
train_z =  optimizer.minimize(z_loss)

# Initializing the variables
mag_grads = tf.gradients(ys = k_energy, xs = Z_old)
grads = tf.gradients(ys = loss, xs = Z)
init = tf.global_variables_initializer()

#grads = tf.gradients(loss, Z)
#grads, _ = tf.clip_by_global_norm(grads, 50) # gradient clipping
#grads_and_vars = list(zip(grads, Z))
#train_op_z = optimizer.apply_gradients(grads_and_vars)


#grads = tf.gradients(ys = loss, xs = Z)
fig  = plt.figure()
x_array = np.array([])
y_array = np.array([])
dir_array = np.array([])
with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver()

	saver.restore(sess, "./model_unbalanced.ckpt")

	batch_x, batch_y = set_batch(batch_size)
	
	for epoch in range(100):
		batch_x, batch_y = set_batch(batch_size)
		batch_x_ = np.interp(batch_x, (0, 1.0), (-0.1, +0.1))
		'''
		randnum = random.randint(1,101)
		if randnum > 50:
			maxx = 4
		else:
			maxx = -4

		batch_x, batch_y = set_batch(batch_size)
	
#		z = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
#		z = np.reshape(np.random.uniform(low=-0.1, high=0.1, size=(3,1)),(-1,1))
	#	_, c = sess.run([train_op, loss], feed_dict={X: batch_x,
                    #                                        Y: batch_y, Z:z})
		print("new batch begins!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		v = 0
		
		for p in range(20):

			z_ol = np.copy(z)
			x_ol = np.copy(batch_x)
			c = sess.run([loss], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z})

			print('loss')
			print(c)
			print("z")
			print(z)
			predy = sess.run([y_], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z})

			print("y_")
			print(predy)
			print("y")
			print(batch_y)
			print("###################################")
			g = sess.run([grads],feed_dict={X: batch_x,Y: batch_y,Z:z})
#			print(g[0][0])
			v_prev = np.copy(v)
			v = 0.001*v - 0.001*g[0][0]
			z += 0.001 * v_prev + (1+0.001)*v
			z = np.clip(z, -0.1, 0.1)			
			
			z_ne = np.copy(z)

			_, z_lo = sess.run([train_z, z_loss], feed_dict={X_old: x_ol, Z_old:z_ol, Z_new: z_ne})


			z_new_pre = sess.run([z_new_pred], feed_dict={X_old: x_ol, Z_old:z_ol})
			print("z loss")
			print(z_lo)
			print("z_old")
			print(z_ol)
			print("z_new")
			print(z_ne)
			print("z_new_pred")
			print(z_new_pre)
			
		#	c = sess.run([loss], feed_dict={X: batch_x,
                     #                                       Y: batch_y, Z:z})
			
			
#			print(c)		
		
		for t in range(1):
			_, c = sess.run([train_op, loss], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z})
		print("saving model")
		saver.save(sess, "./model.ckpt")
		

#		predy = sess.run([y_], feed_dict={X: batch_x,
 #                                                           Y: batch_y, Z:z})	


	#	batch_x_, batch_y_ = set_batch(batch_size,-maxx)
		
		#z_ = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
	#	z = np.reshape(np.array(random.uniform(0, 1)),(-1,1))
		z = np.reshape(np.random.uniform(low=-0.0, high=1.0, size=(100,1)),(-1,1))
		#_, c = sess.run([train_op, loss], feed_dict={X: batch_x_,
                #                                            Y: batch_y_, Z:z_})
		v = 0
		for p in range(1000):
			g = sess.run([grads],feed_dict={X: batch_x_,Y: batch_y_,Z:z})
			v_prev = np.copy(v)
			v = 0.01*v - 0.001*g[0][0]
			z += 0.01 * v_prev + (1+0.01)*v
#			print(type(z))
			z = np.clip(z, -1.0, 1.0)

		for i in range(1):
			_, c = sess.run([train_op, loss], feed_dict={X: batch_x_,
                                                            Y: batch_y_, Z:z})
	#		_, c = sess.run([train_op, loss], feed_dict={X: batch_x,
         #                                                   Y: batch_y, Z:z})
		
		'''
	#uncomment below line 
	#	z = np.reshape(np.random.uniform(low=-0.1, high=0.1, size=(3,1)),(-1,1))
	
		z = np.reshape(np.random.uniform(low=-0.1, high=0.1, size=(3,1)),(-1,1))
#		z = z/np.abs(z) * maxx
	
		c = sess.run([y_], feed_dict={X: batch_x_, Z:z})
	
		for l in range(100):
			energy = np.reshape(sess.run([k_energy], feed_dict={X_old: batch_x_, Z_old:z}),(-1,1))
			print("energy")
			print(energy)
#		g =  sess.run([mag_grads],feed_dict={X_old: batch_x,Z_old:z})
#		v = 0.001*g[0][0]
#		mom = (1+0.001)*v
#		print("mom")
#		print(mom)
		
			print("z")
			print(z)
			c  = sess.run([y_], feed_dict={X: batch_x_, Z:z})
			print("y_pred")
			print(c)
			z_old = np.copy(z)
			z = np.reshape(sess.run([z_new_pred], feed_dict={X_old: batch_x_, Z_old:z}),(-1,1))
#			z = np.clip(z, -0.1, 0.1)
			print("z_new")
			print(z)
			c = sess.run([y_], feed_dict={X: batch_x_, Z:z})
			print("y_pred_new")
			print(c)
			print("z_new_changed")
			if l < 50:
				z_diff = z - z_old
				if np.all(z_diff):
			#	z = z + z_diff/np.abs(z_diff)*0.1*energy
					z = z + z_diff/np.abs(z_diff)*0.1*energy
				z = np.clip(z, -0.1, 0.1)
				print(z)
			c = sess.run([y_], feed_dict={X: batch_x_, Z:z})
			print("x")
			print(batch_x)
			print("y_pred_new_changed")
			print(c)
		

#		print(c)
#		print("random")

		
	#	c = sess.run([loss], feed_dict={X: batch_x_,
         #                                                   Y: batch_y_, Z:z})
#		print(c)


	
	
#	fig = plt.figure()
	
#	x_array = np.array([])
#	y_array = np.array([])
		x_array = np.append(x_array, batch_x)
		y_array = np.append(y_array, c[0][:,0])
	
	'''
	for x in range(10000):
		k = random.uniform(0, 1)
		k_ = np.reshape(k,(-1,1))
	#	z = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
	#	z = np.reshape(np.array(random.uniform(0, 1)),(-1,2))
		z = np.reshape(np.random.uniform(low=-0.1, high=0.1, size=(1,1)),(-1,1))
		y = sess.run([y_],  feed_dict={X:k_,Z:z})
		x_array = np.append(x_array, k)
		print(y[0].shape)
		y_array = np.append(y_array, y[0][0,0])
		print(str(k) + "," + str(y))
	
	'''
	print(y_array.shape)
	ax = fig.add_subplot(111)
	xs = x_array[:]
	ys = y_array[:]
	ax.scatter(xs, ys,color='red',s=1)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')

        #plt.axis('off')
	plt.savefig("./test.png", bbox_inches='tight')
	










