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


learning_rate = 0.0001
training_epochs = 15
batch_size = 2
display_step = 1
num = 0
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

num = 0

def set_batch(batch_size,num):
	x_array = np.array([])
	y_array = np.array([])
#num = 0
	for x in range(batch_size):
		if num == 0:
			maxx = 4
			num = 1
		elif num == 1:
			maxx = -4
			num = 0
	#	randnum = random.randint(1,2)
	#	if randnum == 1:
	#		maxx = 4
	#	if randnum == 2:
	#		maxx = -4
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
	
	return x_array, y_array, num



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

magnitude_real = tf.reduce_sum(tf.abs(Z_new - Z_old),axis = 1, keep_dims = True)

mag_loss = tf.reduce_mean(tf.squared_difference(magnitude_pred,magnitude_real))
z_pred_loss = tf.reduce_mean(tf.squared_difference(z_new_pred , Z_new))

z_loss = z_pred_loss + mag_loss

loss = tf.reduce_mean(tf.squared_difference(y_, Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
train_z =  optimizer.minimize(z_loss)

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
	saver = tf.train.Saver()
#	saver.restore(sess,'./model.ckpt')
	for epoch in range(100000):
		randnum = random.randint(1,101)
		if randnum > 50:
			maxx = 4
		else:
			maxx = -4

		batch_x, batch_y,num = set_batch(3,num)
		batch_x = np.interp(batch_x, (0, 1.0), (-0.1, +0.1))
	
#		z = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
		z = np.reshape(np.random.uniform(low=-0.1, high=0.1, size=(3,1)),(-1,1))
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
			predy = np.reshape(sess.run([y_], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z}),(-1,3))

			print("y_old")
			print(predy)
			print("y")
			print(batch_y)
		
			g = sess.run([grads],feed_dict={X: batch_x,Y: batch_y,Z:z})
#			print(g[0][0])
			v_prev = np.copy(v)
			v = 0.001*v - 0.001*g[0][0]
			z += 0.001 * v_prev + (1+0.001)*v
			z = np.clip(z, -0.1, 0.1)			
			
			z_ne = np.copy(z)
			y_new = np.reshape(sess.run([y_], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z}),(-1,3))
			print("y_new")
			print(y_new)
	#		_, z_lo = sess.run([train_z, z_loss], feed_dict={X_old: x_ol, Z_old:z_ol, Z_new: z_ne})
			
			
			z_new_pre = sess.run([z_new_pred], feed_dict={X_old: x_ol, Z_old:z_ol})

			mag_new =  sess.run([magnitude_real], feed_dict={Z_old:z_ol, Z_new:z_ne})
			mag_pred = sess.run([magnitude_pred], feed_dict={Z_old:z_ol, X_old: x_ol})

			_, z_lo = sess.run([train_z, z_loss], feed_dict={X_old: x_ol, Z_old:z_ol, Z_new: z_ne })
			
			print("z loss")
			print(z_lo)
			print("z_old")
			print(z_ol)
			print("z_new")
			print(z_ne)
			print("z_new_pred")
			print(z_new_pre)
			print("mag")
			print(mag_new )
			print("mag_pred")
			print(mag_pred)		
			print("##########################################################")	
		#	c = sess.run([loss], feed_dict={X: batch_x,
                     #                                       Y: batch_y, Z:z})
			
			
#			print(c)		
		
		for t in range(1):
			_, c = sess.run([train_op, loss], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z})
		print("saving model")
		saver.save(sess, "./model_unbalanced.ckpt")


#		predy = sess.run([y_], feed_dict={X: batch_x,
 #                                                           Y: batch_y, Z:z})	


	#	batch_x_, batch_y_ = set_batch(batch_size,-maxx)
		'''
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
		z = np.reshape(np.random.uniform(low=-0.1, high=0.1, size=(3,1)),(-1,1))
		c = sess.run([loss], feed_dict={X: batch_x,
                                                            Y: batch_y, Z:z})
#		print(c)
#		print("random")

		
	#	c = sess.run([loss], feed_dict={X: batch_x_,
         #                                                   Y: batch_y_, Z:z})
#		print(c)


	


	for x in range(10):
		k = random.uniform(0, 1)
		k_ = np.reshape(k,(-1,1))
	#	z = np.reshape(np.array((random.uniform(0, 1),random.uniform(0, 1))),(-1,2))
		z = np.reshape(np.array(random.uniform(0, 1)),(-1,2))
		y = sess.run([y_],  feed_dict={X:k_,Z:z})
		print(str(k) + "," + str(y))













