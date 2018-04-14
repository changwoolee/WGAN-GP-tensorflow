import tensorflow as tf



def batch_norm(x, epsilon=1e-5, decay=0.9, name='batch_norm'):
	return tf.contrib.layers.batch_norm(x, decay=decay, epsilon=epsilon, scale=True)



def linear(name, x, output_dim, stddev=0.02):
	with tf.variable_scope(name):
		w = tf.get_variable('w', shape=[x.get_shape()[-1], output_dim], 
				initializer=tf.random_normal_initializer(stddev=stddev))
		y = tf.matmul(x, w)
		return y


def deconv2d(name, input_, output_shape, strides=[1,2,2,1],  ksize=4, stddev=0.02, trainable=True):
	with tf.variable_scope(name):
		w = tf.get_variable('w', shape=[ksize, ksize, output_shape[-1], input_.get_shape()[-1]], 
				initializer=tf.random_normal_initializer(stddev=stddev), trainable=trainable)
		deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
				strides=strides)

		return deconv


def conv2d(name, input_, output_dim, strides=[1,2,2,1], ksize=4, stddev=0.02):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [ksize, ksize, input_.get_shape()[-1],output_dim],
				initializer=tf.random_normal_initializer(stddev=stddev))

		conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME')
		return conv



def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)
