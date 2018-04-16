from __future__ import division
import tensorflow as tf
import os
from ops import *
from glob import glob
from utils import *
import numpy as np


class WGAN(object):
	def __init__(self, sess, input_height=108, input_width=108, crop=True,
			batch_size=64, sample_num=64, output_height=64, output_width=64,
			y_dim=None, z_dim=100, g_dim=None, d_dim=None, c_dim=3,
			dataset_name='celebA', input_fname_pattern='*.jpg',
			log_dir=None, 
			max_iterations=1e5,
			n_critic=5, lr=2e-5, beta1=0., beta2=0.9):

		"""
		Original code from DCGAN-tensorflow by carpedm20
		"""


		self.sess = sess
		self.input_height = input_height
		self.input_width = input_width
		self.crop = crop
		self.batch_size = batch_size
		self.sample_num= sample_num
		self.output_height=output_height
		self.output_width = output_width
		self.z_dim = z_dim
		
		self.c_dim = c_dim
		self.g_dim = g_dim
		self.d_dim = d_dim
		self.dataset_name = dataset_name
		self.input_fname_pattern = input_fname_pattern
		self.log_dir = log_dir
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))		
		if len(self.data) is 0:
			raise Exception("[!] No training data. Program shut down")

		self.max_iterations = max_iterations
		self.n_critic=n_critic
		self.lr = lr #learning rate
		self.beta1=beta1
		self.beta2=beta2



		self.build_model()

	def read_input(self):
		"""
			Code from https://github.com/tdeboissiere/DeepLearningImplementations
		"""
		reader = tf.WholeFileReader()
		filename_queue = tf.train.string_input_producer(self.data)
		key, value = reader.read(filename_queue)
		image = tf.image.decode_jpeg(value, channels=self.c_dim, name="dataset_image")

		# Crop and other random augmentations
		image = tf.image.random_flip_left_right(image)

		image = tf.image.crop_to_bounding_box(image, (218 - self.input_height) //2, (178 - self.input_width) // 2, self.input_height, self.input_width)
		if self.crop:
			image = tf.image.resize_images(image, [self.output_height, self.output_width], method=tf.image.ResizeMethod.BICUBIC)

		image = tf.to_float(image) / 127.5 - 1.

		num_preprocess_threads=4
		num_examples_per_epoch=800
		min_queue_examples = int(0.1 * num_examples_per_epoch)
		img_batch = tf.train.batch([image],
									batch_size=self.batch_size,
									num_threads=4,
									capacity=min_queue_examples + 2*self.batch_size)

		return img_batch
		


	def build_model(self):
		if self.crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_width, self.c_dim]

		self.X_real = self.read_input()

		self.z = tf.placeholder(
				tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.summary.histogram("z", self.z)

		self.X_fake = self.generator(self.z)
		self.real_img_sum = tf.summary.image("image_real", self.X_real, max_outputs=4)
		self.fake_img_sum = tf.summary.image("image_fake", self.X_fake, max_outputs=4)

		self.d_logits_fake = self.discriminator(self.X_fake, reuse=False)
		self.d_logits_real = self.discriminator(self.X_real, reuse=True)
		print(self.d_logits_fake.get_shape())
		# WGAN Loss
		self.d_loss = tf.reduce_mean(self.d_logits_fake) - tf.reduce_mean(self.d_logits_real)
		self.g_loss = -tf.reduce_mean(self.d_logits_fake)

		# Gradient Penalty
		self.epsilon = tf.random_uniform(
				shape=[self.batch_size, 1, 1, 1],
				minval=0.,
				maxval=1.)
		X_hat = self.X_real + self.epsilon * (self.X_fake - self.X_real)
		D_X_hat = self.discriminator(X_hat, reuse=True)
		grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
		red_idx = [1]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.d_loss = self.d_loss + 10.0 * gradient_penalty

		self.d_loss_sum = tf.summary.scalar("Discriminator_loss", self.d_loss)
		self.g_loss_sum = tf.summary.scalar("Generator_loss", self.g_loss)
		self.gp_sum = tf.summary.scalar("Gradient_penalty", gradient_penalty)

		train_vars = tf.trainable_variables()

		for v in train_vars:
			tf.add_to_collection("reg_loss", tf.nn.l2_loss(v))

		self.generator_vars = [v for v in train_vars if 'g_' in v.name]
		self.discriminator_vars = [v for v in train_vars if 'd_' in v.name]



		self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='g_opt',
				beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=self.generator_vars)
		self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='d_opt',
				beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=self.discriminator_vars)

		self.d_sum = tf.summary.merge([self.z_sum, self.d_loss_sum, self.gp_sum, self.real_img_sum, self.fake_img_sum])
		self.g_sum = tf.summary.merge([self.z_sum, self.g_loss_sum, self.fake_img_sum])
		self.saver = tf.train.Saver()
		self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

		
		self.initialize_model()

	def initialize_model(self):
		print("[*] initializing network...")

		self.sess.run(tf.global_variables_initializer())
#		ckpt = tf.train.get_checkpoint_state(self.log_dir)
#		if ckpt and ckpt.model_checkpoint_path:
#			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
#			print("[*] Model restored.")
		_, self.counter = self.load(self.log_dir)
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(self.sess, self.coord)


	def train(self):
		print("[*] Training Improved Wasserstein GAN")
		counter=self.counter
		global_step = tf.Variable(tf.constant(0), trainable=False, name='global_step')
		tf.add_to_collection('global_step', global_step)

		sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
		start_time = time.time()
		
		for step in xrange(1, int(self.max_iterations)):
			batch_idxs = len(self.data)
			# Critic
			for critic_iter in range(self.n_critic):
				self.batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
				# Update Discriminator
				_, summary_str = self.sess.run([self.d_optimizer, self.d_sum], feed_dict={self.z: self.batch_z})#, self.X_real: self.batch_images}) 
			# Update Generator
			self.batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
			self.summary_writer.add_summary(summary_str, counter)
			_, summary_str = self.sess.run([self.g_optimizer, self.g_sum] ,feed_dict={self.z: self.batch_z})
			self.summary_writer.add_summary(summary_str, counter)
			print(counter)
#if counter%100==0:
#				summary_str = self.sess.run(self.summary_op, feed_dict={self.z: sample_z})#, self.X_real: sample_images})
#				self.summary_writer.add_summary(summary_str, counter)
				
			if counter%200==0:
				stop_time = time.time()
				duration = (stop_time - start_time) / 200.0
				start_time = stop_time
				g_loss_val, d_loss_val = self.sess.run([self.g_loss, self.d_loss], feed_dict={self.z: self.batch_z})#, self.X_real: self.batch_images})
				print("Time: %g/itr, Step: %d, generator loss: %g, discriminator loss: %g" % (duration, counter, g_loss_val, d_loss_val))

			if counter%5000==0:
				global_step=counter
				self.saver.save(self.sess, self.log_dir + "/model.ckpt", global_step=counter)

				
			counter+=1

				
		

	def generator(self, z):
		with tf.variable_scope("generator") as scope:
			dims = [self.g_dim*8, self.g_dim*4, self.g_dim*2, self.g_dim, 3]
			s_h, s_w = self.output_height, self.output_width
			s_h, s_w = [s_h//16, s_h//8, s_h//4, s_h//2, s_h], [s_w//16, s_w//8, s_w//4, s_w//2, s_w]
	
			self.z_ = linear("g_h0_lin", z, dims[0] * s_h[0] * s_w[0])

			self.h0 = tf.reshape(self.z_, [-1, s_h[0], s_w[0], dims[0]])
			h0 = tf.nn.relu(batch_norm(self.h0, name="g_bn0"))


			h1 = deconv2d("g_h1", h0, [self.batch_size, s_h[1], s_w[1], dims[1]])
			h1 = tf.nn.relu(batch_norm(h1, name="g_bn1"))
		
			h2 = deconv2d("g_h2", h1, [self.batch_size, s_h[2], s_w[2], dims[2]])
			h2 = tf.nn.relu(batch_norm(h2, name="g_bn2"))

			h3 = deconv2d("g_h3", h2, [self.batch_size, s_h[3], s_w[3], dims[3]])
			h3 = tf.nn.relu(batch_norm(h3, name="g_bn3"))

			h4 = deconv2d("g_h4", h3, [self.batch_size, s_h[4], s_w[4], dims[4]])
		
			return tf.nn.tanh(h4, name='pred_image')

		
	def discriminator(self, input_image, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			
			dims = [self.c_dim, self.d_dim, self.d_dim*2, self.d_dim*4, self.d_dim*8]
#		s_h, s_w = self.output_height, self.output_width
#		s_h, s_w = [s_h, s_h*2, s_h*4, s_h*8, s_h*16], [s_w, s_w*2, s_w*4, s_w*8, s_w*16]

			h0 = conv2d("d_h0", input_image, dims[1])
			h0 = lrelu(batch_norm(h0, name="d_bn0"))
		
			h1 = conv2d("d_h1", h0, dims[2])
			h1 = lrelu(batch_norm(h1, name="d_bn1"))
		
			h2 = conv2d("d_h2", h1, dims[3])
			h2 = lrelu(batch_norm(h2, name="d_bn2"))

			h3 = conv2d("d_h3", h2, dims[4])
			h3 = lrelu(batch_norm(h3, name="d_bn3"))
			
		
			h3 = tf.reshape(h3, [-1, 4*4*self.d_dim*8])
			h_pred = linear("d_h4", h3, 1)
			h_pred = tf.reshape(h_pred, [-1])
			return h_pred
		
		
	def load(self, log_dir):
		print("[*] Reading Checkpoints...") 
		ckpt = tf.train.get_checkpoint_state(log_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			global_step = tf.get_collection_ref('global_step')[0]
			gs = self.sess.run(global_step)
			print("[*] Model restored.")
			return True, gs
		else:
			print("[*] Failed to find a checkpoint")
			return False, 0
		


