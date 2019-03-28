"""
The solver for training
"""

import tensorflow as tf
import numpy as np

import logging
from time import time
from os import path, makedirs

from model import *
from train400_data import *
from ops import *
from utils import *


flags = tf.app.flags
conf = flags.FLAGS


class Solver(object):
	def __init__(self):
		# path
		self.data_dir = conf.data_dir
		self.models_dir = conf.models_dir
		self.logFilename = conf.log_name
		
		# make dirs
		if not path.exists(self.models_dir):
			makedirs(self.models_dir)
		
		# soft constraint for total epochs
		self.num_epoch = conf.num_epoch + 1
		
		# hyper parameters
		self.batch_size = conf.batch_size
		self.weight_decay = conf.weight_decay
		
		# learning rate
		self.lr = tf.placeholder(tf.float32)
		self.base_lr = conf.base_lr
		self.power = conf.power
		self.end_lr = conf.end_lr
		
		# resuming and finetune
		self.resume = conf.resume
		self.finetune = conf.finetune
		
		if conf.iters == None:
			if self.resume or self.finetune:
				raise ValueError
		
		# get datas and labels for training
		dataset = Train400_Data(filename=path.join(self.data_dir, 'train400.tfrecord'),
	                            num_epoch=self.num_epoch, sigma=50, batch_size=self.batch_size, scope='train400')		
		
		with tf.device('/gpu:0'):
			# build the inference graph
			net = Net(data=dataset.datas, label=dataset.labels, wl=self.weight_decay)
			net.build_net()
			
			# create an optimizer that performs gradient descent
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(net.total_loss)
			self.total_loss = net.total_loss
	
	
	def init_logging(self):
		logging.basicConfig(
		    level    = logging.DEBUG,
		    #format   = 'LINE %(lineno)-4d  %(levelname)-8s %(message)s',
		    format   = '%(message)s',
		    datefmt  = '%m-%d %H:%M',
		    filename = self.logFilename,
		    filemode = 'w')
		
		# define a Handler which writes INFO messages or higher to the sys.stderr
		console = logging.StreamHandler()
		console.setLevel(logging.DEBUG)
		
		# set a format which is simpler for console use
		#formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s');
		formatter = logging.Formatter('%(message)s')
		# tell the handler to use this format
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)
	
	
	def train(self, disp_freq, save_freq, summary_freq):
		# initialize logging
		self.init_logging()
		
		# operations for initialization
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver(max_to_keep=int(10e3))
		
		# create a session for running operations in the graph
		config = tf.ConfigProto(allow_soft_placement=True)
		# config.gpu_options.allow_growth = True
		config.gpu_options.allow_growth = False
		#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 # enable XLA
		sess = tf.Session(config=config)
		
		# initialize the variables (like the epoch counter)
		sess.run(init_op)
		
		# restore trained weights for resuming
		if self.resume or self.finetune:
			saver.restore(sess, path.join(conf.finetune_models_dir, 'model.ckpt-' + str(conf.iters)))
		
		summary_writer = tf.summary.FileWriter(self.models_dir, sess.graph)
		
		# start input enqueue threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		# global iterations for resuming
		if self.resume:
			iters = conf.iters
		
		# for training and finetune
		else:
			iters = 0
		
		# accumulational variables
		sum_time = 0
		sum_loss = 0
		
		# trace options and metadata
		checkpoint_path = path.join(self.models_dir, 'model.ckpt')
		# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		# run_metadata = tf.RunMetadata()
		
		# save iteration 0 and metagraph
		saver.save(sess, checkpoint_path, global_step=iters)
		
		# generate summary
		# summary_str = sess.run(summary_op, options=run_options, run_metadata=run_metadata)
		# summary_writer.add_run_metadata(run_metadata, 'step%03d' % iters)
		# summary_writer.add_summary(summary_str, iters)
		
		# decay policy of learning rate
		decay_fraction_of_epochs = 1.0
		self.decay_steps = (num_examples_per_epoch_for_train * self.num_epoch * decay_fraction_of_epochs) // self.batch_size
		
		total_loss = sess.run(self.total_loss, feed_dict={self.lr: self.base_lr})
		logging.info('step %d, loss = %.8f' % (iters, total_loss))
		
		try:
			# training loop
			while not coord.should_stop():
				# calculate current learning rate (truncated polynomial decay)
				if iters <= self.decay_steps:
					current_lr = (self.base_lr - self.end_lr) * pow((1 - float(iters) / self.decay_steps), (self.power)) + self.end_lr
				else:
					current_lr = self.end_lr
				
				# run training steps or whatever
				t1 = time()
				_, total_loss = sess.run([self.train_op, self.total_loss], feed_dict={self.lr: current_lr})
				t2 = time()
				iters += 1
				
				# accumulate
				sum_time += t2 - t1
				sum_loss += total_loss
				
				# display
				if iters % disp_freq == 0:
					logging.info('step %d, loss = %.4f (lr: %.8f, time: %.2fs)'
					             % (iters, sum_loss / disp_freq, current_lr, sum_time))
					sum_time = 0
					sum_loss = 0
				
				# save checkpoint
				if iters % save_freq == 0:
					saver.save(sess, checkpoint_path, global_step=iters, write_meta_graph=False)
				
				# write summary
				if iters % summary_freq == -1:
					summary_str = sess.run(summary_op, options=run_options, run_metadata=run_metadata)
					summary_writer.add_run_metadata(run_metadata, 'step%03d' % iters)
					summary_writer.add_summary(summary_str, iters)
				
		except tf.errors.OutOfRangeError:
			logging.info('Done training -- epoch limit reached')
		finally:
			# when done, ask the threads to stop
			coord.request_stop()
		
		# wait for threads to finish
		coord.join(threads)
		sess.close()
