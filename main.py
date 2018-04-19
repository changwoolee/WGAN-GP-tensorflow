import os
import scipy.misc
import numpy as np

import tensorflow as tf

from model import WGAN
from utils import pp, visualize, show_all_variables, forward_test

flags = tf.app.flags
flags.DEFINE_integer("max_epoch",150,"Maximum epoch")
flags.DEFINE_integer("input_height",108,"Input height")
flags.DEFINE_integer("input_width",108,"Input width")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("z_dim", 100, "Dimension of z")
flags.DEFINE_integer("output_height", 64, "Output height")
flags.DEFINE_integer("output_width", 64, "Output width")
flags.DEFINE_boolean("crop",True,"Center Crop")
flags.DEFINE_string("dataset","celebA","Name of dataset")
flags.DEFINE_string("data_pattern","*.jpg","data file pattern")
flags.DEFINE_string("log_dir","./logs","Log directory path")
flags.DEFINE_string("sample_dir","./samples","sample directory")
flags.DEFINE_integer("n_critic",5,"Number of critic iteration")
flags.DEFINE_float("beta1",0.,"beta1 for Adam Optimizer")
flags.DEFINE_float("beta2",0.9,"beta2 for Adam Optimizer")
flags.DEFINE_float("learning_rate",1e-4,"learning rate")
flags.DEFINE_integer("g_dim",64,"Dimension of generator")
flags.DEFINE_integer("d_dim",64,"Dimension of discriminator")
flags.DEFINE_boolean("train",False,"train")
flags.DEFINE_boolean("forward_test",False,"Forward Test")
FLAGS = flags.FLAGS


def main(_):
	pp.pprint(flags.FLAGS.__flags)

#	run_config = tf.ConfigProto()
#	run_config.gpu_options.allow_growth=True

#	with tf.Session(config=run_config) as sess:
	with tf.Session() as sess:
		wgan = WGAN(sess,
				input_height=FLAGS.input_height,
				input_width=FLAGS.input_width,
				crop=FLAGS.crop,
				batch_size=FLAGS.batch_size,
				output_height=FLAGS.output_height,
				output_width=FLAGS.output_width,
				z_dim=FLAGS.z_dim,
				g_dim=FLAGS.g_dim,
				d_dim=FLAGS.d_dim,
				dataset_name=FLAGS.dataset,
				input_fname_pattern=FLAGS.data_pattern,
				log_dir=FLAGS.log_dir,
				sample_dir=FLAGS.sample_dir,
				max_epoch=FLAGS.max_epoch,
				n_critic=FLAGS.n_critic,
				lr=FLAGS.learning_rate,
				beta1=FLAGS.beta1,
				beta2=FLAGS.beta2)


		show_all_variables()

		if FLAGS.train:
			wgan.train()
		else:
			if not wgan.load(FLAGS.checkpoint_dir)[0]:
				raise Exception("[!] Train a model first, then run test mode")



		if FLAGS.forward_test:
			forward_test(sess,wgan,FLAGS, FLAGS.test_num)
		OPTION = 1
		visualize(sess.wgan, FLAGS, OPTION)


if __name__=='__main__':
	tf.app.run()
