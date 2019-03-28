"""
Instantiate a solver for test
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matlab.engine
import os

from ops import *
from utils import *


def modcrop(im, modulo):
	if len(im.shape) == 3:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1], :]
	elif len(im.shape) == 2:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1]]
	else:
		raise AttributeError
	return im


def shave(im, border):
	if len(im.shape) == 3:
		return im[border[0] : -border[0], 
			      border[1] : -border[1], :]
	elif len(im.shape) == 2:
		return im[border[0] : -border[0], 
			      border[1] : -border[1]]
	else:
		raise AttributeError


def dfus_block(bottom, i):
	act = tf.nn.relu

	with tf.name_scope('dfus_block' + str(i)):
		conv1  = act(conv2d(bottom, 24, [1, 1], wl=None, scope='conv' + str(i) + '_i'), name='relu' + str(i) + '_i')

		feat1  = act(conv2d(conv1, 6, [3, 3], scope='conv' + str(i) + '_1'), name='relu' + str(i) + '_1')
		feat15 = act(conv2d(feat1, 3, [3, 3], dilated=2, scope='conv' + str(i) + '_15'), name='relu' + str(i) + '_15')

		feat2  = act(conv2d(conv1, 6, [3, 3], dilated=2, scope='conv' + str(i) + '_2'), name='relu' + str(i) + '_2')
		feat23 = act(conv2d(feat2, 3, [3, 3], scope='conv' + str(i) + '_23'), name='relu' + str(i) + '_23')

		feat = tf.concat([feat1, feat15, feat2, feat23], 1, name='conv' + str(i) + '_c1')
		feat = act(conv2d(feat, 8, [1, 1], wl=None, scope='conv' + str(i) + '_r'), name='relu' + str(i) + '_r')

		top = tf.concat([bottom, feat], 1, name='conv' + str(i) + '_c2')

	return top


def ddfn(bottom, step):
	act = tf.nn.relu

	with tf.variable_scope('ddfn_' + str(step)):
		with tf.name_scope('msfeat'):
			conv13  = act(conv2d(bottom, 8, [3, 3], scope='conv1_3'), name='relu1_3')
			conv15  = act(conv2d(bottom, 8, [3, 3], dilated=2, scope='conv1_5'), name='relu1_5')

			conv133 = act(conv2d(conv13, 6, [3, 3], dilated=2, scope='conv1_3_3'), name='relu1_3_3')
			conv153 = act(conv2d(conv15, 6, [3, 3], scope='conv1_5_3'), name='relu1_5_3')

			conv1 = tf.concat([conv13, conv15, conv133, conv153], 1, name='conv1_c')

		feat = dfus_block(conv1, 2)

		for i in range(3, 10, 1):
			feat = dfus_block(feat, i)

		top = conv2d(feat, 1, [1, 1], W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
	                 add_biases=False, wl=None, scope='convr')

		return top


def build_net(im):
	with tf.name_scope('net'):
		output = ddfn(im, 1)
	return output
	

def main():
	# folder path
	folder = '../datas/Set12'
	# folder = '../datas/BSD68'
	
	# sigma
	sigma = 50.0
	
	# generate the file list
	filepath = os.listdir(folder)
	filepath.sort()
	
	# recreate the network
	im_input = tf.placeholder('float', [1, 1, None, None])
	with tf.device('/gpu:0'):
		output = build_net(im_input)
		
	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	# restore weights
	saver = tf.train.Saver()
	# 27.100 (Set12) 26.188 (BSD68)
	saver.restore(sess, os.path.join('./models', 'model.ckpt'))
	sum_psnr = 0
	for i in  np.arange(0, len(filepath), 1):
		im = np.array(Image.open(os.path.join(folder, filepath[i])))
		im = modcrop(im, 2)
		# (Image.fromarray(im)).show()
		
		im = im.astype(np.float32) / 255.0
		im = np.expand_dims(im, axis=0)
		im = np.expand_dims(im, axis=0)
		
		#np.random.seed(0)
		#noise = np.random.normal(loc=0.0, scale=sigma / 255.0, size=im.shape)
		
		_ = eng.rng(0, 'v4')
		eng.workspace['sigma'] = sigma
		noise = eng.eval("(sigma / 255) * randn(%d, %d)" % ((np.squeeze(im)).shape[0], (np.squeeze(im)).shape[1]))
		noise = np.asarray(noise, dtype=np.float32)			
		im_n = im + noise
		
		im_dn = sess.run(output, feed_dict={im_input: im_n})
		
		im = np.squeeze(im)
		im_dn = np.squeeze(im_dn)
		
		im = shave(im, [1, 1])
		im_dn = shave(im_dn, [1, 1])
				
		psnr = compute_psnr(im * 255, im_dn * 255)
		print('%s : %.4f dB' % (filepath[i], psnr))
		sum_psnr += psnr

		im_dn[im_dn>1] = 1
		im_dn[im_dn<0] = 0
	
	avg_psnr = sum_psnr / len(filepath)
	print('Averaged PSNR: %.4f dB' % avg_psnr)
	

if __name__ == '__main__':
	eng = matlab.engine.start_matlab()
	main()
	eng.quit()
