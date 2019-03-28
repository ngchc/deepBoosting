"""
The core Boosting Network Model
"""

import tensorflow as tf
import numpy as np
from ops import *


class Net(object):
	def __init__(self, data, label, wl):
		"""
		Args: [0, 1]
		  data : [batch_size, height, width, channels] float32
		  label: [batch_size, height, width, channels] float32
		"""
		# training dataset
		self.data = data
		self.label = label
		self.batchsize = self.data.get_shape().as_list()[0]
		
		# parameter lists for weights and biases
		self.W_params = []
		self.b_params = []
		
		# coefficient of weight decay
		self.wl = wl
	
	
	def _check_shape(self, a, b, scale=1):
		N1, C1, H1, W1 = a.get_shape().as_list()
		N2, C2, H2, W2 = b.get_shape().as_list()
		
		assert N1 == N2, 'Inequality of batchs!'
		assert C1 == C2, 'Inequality of channels!'
		assert H1 == H2 / scale, 'Inequality of heights!'
		assert W1 == W2 / scale, 'Inequality of widths!'
	
	
	def dfus_block(self, bottom, i):
		act = tf.nn.relu
		
		with tf.name_scope('dfus_block' + str(i)):
			conv1  = act(conv2d(bottom, 24, [1, 1], wl=None, scope='conv' + str(i) + '_i'), name='relu' + str(i) + '_i')
			
			feat1  = act(conv2d(conv1, 6, [3, 3], wl=self.wl, scope='conv' + str(i) + '_1'), name='relu' + str(i) + '_1')
			feat15 = act(conv2d(feat1, 3, [3, 3], dilated=2, wl=self.wl, scope='conv' + str(i) + '_15'), name='relu' + str(i) + '_15')
			
			feat2  = act(conv2d(conv1, 6, [3, 3], dilated=2, wl=self.wl, scope='conv' + str(i) + '_2'), name='relu' + str(i) + '_2')
			feat23 = act(conv2d(feat2, 3, [3, 3], wl=self.wl, scope='conv' + str(i) + '_23'), name='relu' + str(i) + '_23')
			
			feat = tf.concat([feat1, feat15, feat2, feat23], 1, name='conv' + str(i) + '_c1')
			feat = act(conv2d(feat, 8, [1, 1], wl=None, scope='conv' + str(i) + '_r'), name='relu' + str(i) + '_r')
			
			top = tf.concat([bottom, feat], 1, name='conv' + str(i) + '_c2')
		
		return top
	
	
	def ddfn(self, bottom, step):
		act = tf.nn.relu
		
		with tf.variable_scope('ddfn_' + str(step)):
			with tf.name_scope('msfeat'):
				conv13  = act(conv2d(bottom, 8, [3, 3], wl=self.wl, scope='conv1_3'), name='relu1_3')
				conv15  = act(conv2d(bottom, 8, [3, 3], dilated=2, wl=self.wl, scope='conv1_5'), name='relu1_5')
				
				conv133 = act(conv2d(conv13, 6, [3, 3], dilated=2, wl=self.wl, scope='conv1_3_3'), name='relu1_3_3')
				conv153 = act(conv2d(conv15, 6, [3, 3], wl=self.wl, scope='conv1_5_3'), name='relu1_5_3')
				
				conv1 = tf.concat([conv13, conv15, conv133, conv153], 1, name='conv1_c')
			
			feat = self.dfus_block(conv1, 2)
			
			for i in range(3, 10, 1):
				feat = self.dfus_block(feat, i)
			
			top = conv2d(feat, 1, [1, 1], W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
			             add_biases=False, wl=None, scope='convr')
			
			return top
	
	
	def build_net(self, summary=False):
		with tf.name_scope('net'):
			outputs = self.ddfn(self.data, 1)
			
			# crop the boundary
			outputs = tf.image.crop_to_bounding_box(tf.transpose(outputs, [0, 2, 3, 1]), 1, 1, 48, 48)
			labels = tf.image.crop_to_bounding_box(tf.transpose(self.label, [0, 2, 3, 1]), 1, 1, 48, 48)
			
			# mean square error
			self.l2_loss = (1.0 / self.batchsize) * tf.nn.l2_loss(outputs - labels)
			tf.add_to_collection('losses', self.l2_loss)
			
			# total loss collected from 'losses'
			self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
			
			# summary data and label images
			if summary:
				tf.summary.image('data', self.data, max_outputs=1)
				tf.summary.image('label', labels, max_outputs=1)
				tf.summary.image('output', outputs, max_outputs=1)
