"""
Various tensorflow utilities
"""

import tensorflow as tf
import numpy as np


def atrous_conv2d(value, filters, rate, padding, name=None):
	return tf.nn.convolution(
		input=value,
		filter=filters,
		padding=padding,
		dilation_rate=np.broadcast_to(rate, (2,)),
	    data_format='NCHW',
	    name=name)


def conv2d(inputs, num_outputs, kernel_shape=[3, 3], strides=[1, 1], add_biases=True, pad='SAME', dilated=1, reuse=False,
           W_init=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), # msra
           b_init=tf.constant_initializer(0.0), W_params=None, b_params=None, wl=None, wl_type=tf.nn.l2_loss, summary=False, scope='conv2d'):
	"""
	Args:
	  inputs: NCHW
	  num_outputs: the number of filters
	  kernel_shape: [height, width]
	  strides: [height, width]
	  pad: 'SAME' or 'VALID'
	  W/b_params: lists for layer-wise learning rate and gradient clipping
	  wl: add weight losses to collection
	  reuse: reusage of variables
	  dilated: convolution with holes
	
	Returns:
	  outputs: NCHW
	"""
	with tf.variable_scope(scope, reuse=reuse):
		# get shapes
		kernel_h, kernel_w = kernel_shape
		stride_h, stride_w = strides
		batch_size, in_channel, height, width = inputs.get_shape().as_list()
		
		weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
		weights = tf.get_variable('w', weights_shape, tf.float32, W_init)
		
		# add summary for w
		if summary and not reuse:
			tf.summary.histogram('hist_w', weights)
		
		# add to the list of weights
		if W_params is not None and not reuse:
			W_params += [weights]
		
		# 2-D convolution
		if dilated != 1:
			outputs = atrous_conv2d(inputs, weights, rate=dilated, padding=pad)
		else:
			outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding=pad, data_format='NCHW')
		
		# add biases
		if add_biases:
			biases = tf.get_variable('b', [num_outputs], tf.float32, b_init)
			
			# add summary for b
			if summary and not reuse:
				tf.summary.histogram('hist_b', biases)
				
			# add to the list of biases
			if b_params is not None and not reuse:
				b_params += [biases]
			
			outputs = tf.nn.bias_add(outputs, biases, data_format='NCHW')
		
		# add weight decay
		if wl is not None:
			weight_loss = tf.multiply(wl_type(weights), wl, name='weight_loss')
			tf.add_to_collection('losses', weight_loss)
		
		return outputs


def _cal_output_shape(height, width, kernel_shape, strides, pad):
	"""
	Args:
	  kernel_shape: [height, width]
	  strides: [height, width]
	  pad: 'SAME' or 'VALID'
	
	Returns:
	  output_shape: [output_h, output_w]
	"""
	# check the options of padding
	if (pad != 'SAME' and pad != 'VALID'):
		raise Exception("the option of padding must be either 'SAME' or 'VALID'")

	kernel_h, kernel_w = kernel_shape
	stride_h, stride_w = strides

	# calculate output_h
	if strides[0] == 1:
		if pad == 'SAME':
			output_h = height
		else:	# VALID
			output_h = height + kernel_h - 1
	elif strides[0] > 1:
		if pad == 'SAME':
			# output_h = (height - 1) * stride_h + kernel_h - 2 * pad_h
			# assuming padding is 1
			output_h = (height - 1) * stride_h + kernel_h - 2
		else:	# VALID 
			output_h = (height - 1) * stride_h + kernel_h
	else:
		raise ValueError

	# calculate output_w
	if strides[1] == 1:
		if pad == 'SAME':
			output_w = width
		else:	# VALID
			output_w = width + kernel_w - 1
	elif strides[1] > 1:
		if pad == 'SAME':
			output_w = (width - 1) * stride_w + kernel_w - 2
		else:	# VALID
			output_w = (width - 1) * stride_w + kernel_w
	else:
		raise ValueError
	
	return output_h, output_w


def conv2d_transpose(inputs, num_outputs, kernel_shape=[3, 3], strides=[1, 1], add_biases=True, pad='VALID', reuse=False,
                     W_init=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), # msra
                     b_init=tf.constant_initializer(0.0), W_params=None, b_params=None, wl=None, wl_type=tf.nn.l2_loss, 
                     summary=True, scope='conv2d_transpose'):
	"""
	Args:
	  inputs: NCHW
	  num_outputs: the number of filters
	  kernel_shape: [height, width]
	  strides: [height, width]
	  W/b_params: lists for layer-wise learning rate and gradient clipping
	  wl: add weight losses to collection
	  reuse: reusage of variables
	
	Returns:
	  outputs: NCHW
	"""
	with tf.variable_scope(scope, reuse=reuse):
		# get shapes
		kernel_h, kernel_w = kernel_shape
		stride_h, stride_w = strides		
		batch_size, in_channel, height, width = inputs.get_shape().as_list()
		
		# in_channel <--> num_outputs
		weights_shape = [kernel_h, kernel_w, num_outputs, in_channel]
		weights = tf.get_variable('w', weights_shape, tf.float32, W_init)
		
		# add summary
		if summary and not reuse:
			tf.summary.histogram('hist_w', weights)
		
		# add to the list of weights
		if W_params is not None and not reuse:
			W_params += [weights]

		# calculate output shape
		output_h, output_w = _cal_output_shape(height, width, kernel_shape, strides, pad)
		
		# the transpose of 2-D convolution
		output_shape = tf.constant([batch_size, output_h, output_w, num_outputs])
		outputs = tf.nn.conv2d_transpose(inputs, weights, output_shape, [1, stride_h, stride_w, 1], padding=pad, data_format='NCHW')
		
		# add biases
		if add_biases:
			biases = tf.get_variable('b', [num_outputs], tf.float32, b_init)
			
			# add summary for b
			if summary and not reuse:
				tf.summary.histogram('hist_b', biases)
				
			# add to the list of biases
			if b_params is not None and not reuse:
				b_params += [biases]
			
			outputs = tf.nn.bias_add(outputs, biases, data_format='NCHW')
		
		# add weight decay
		if wl is not None:
			weight_loss = tf.multiply(wl_type(weights), wl, name='weight_loss')
			tf.add_to_collection('losses', weight_loss)
		
		return outputs


def batch_norm(x, train=True, reuse=False, scope=None):
	"""
	center: add offset of beta to normalized tensor
	scale:  multiply by gamma
	trainable: add variables to GraphKeys.TRAINABLE_VARIABLES
	updates_collections: whether to force the updates in place
	"""
	return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None,
	                                    trainable=True, is_training=train, reuse=reuse, scope=scope)


def resnet_block(inputs, num_outputs, kernel_shape=[3, 3], strides=[1, 1], W_params=None, b_params=None, reuse=False,
                 W_init=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), # msra
                 b_init=tf.constant_initializer(0.0), act_func=tf.nn.relu, wl=None, wl_type=tf.nn.l2_loss, summary=True, train=True, scope=None):
	"""
	Args:
	  inputs: NCHW
	  num_outputs: the number of filters
	  kernel_shape: [kernel_h, kernel_w]
	  strides: [height, width]
	  reuse: reusage of variables
	
	Returns:
	  outputs: NCHW
	"""
	with tf.variable_scope(scope, reuse=reuse):
		conv1 = conv2d(inputs, num_outputs, kernel_shape, strides, add_biases=False, reuse=reuse,
		               W_init=W_init, b_init=b_init, W_params=W_params, b_params=b_params,
		               wl=wl, wl_type=wl_type, summary=summary, scope='conv1')
		bn1 = batch_norm(conv1, train=train, reuse=reuse, scope='bn1')
		relu1 = act_func(bn1)
		
		conv2 = conv2d(relu1, num_outputs, kernel_shape, strides, add_biases=False, reuse=reuse,
		               W_init=W_init, b_init=b_init, W_params=W_params, b_params=b_params,
		               wl=wl, wl_type=wl_type, summary=summary, scope='conv2')
		bn2 = batch_norm(conv2, train=train, reuse=reuse, scope='bn2')
		output = inputs + bn2

		return output
