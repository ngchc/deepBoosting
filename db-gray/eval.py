import os
import numpy as np
import tensorflow as tf
from PIL import Image


def modcrop(im, modulo):
	if len(im.shape) == 3:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1], :]
	elif len(im.shape) == 2:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1]]
	else: raise AttributeError
	return im


def shave(im, border):
	if len(im.shape) == 3:
		return im[border[0] : -border[0], 
			      border[1] : -border[1], :]
	elif len(im.shape) == 2:
		return im[border[0] : -border[0], 
			      border[1] : -border[1]]
	else: raise AttributeError


def compute_psnr(im1, im2):
	if im1.shape != im2.shape:
		raise Exception('the shapes of two images are not equal')
	rmse = np.sqrt(((np.asfarray(im1) - np.asfarray(im2)) ** 2).mean())
	psnr = 20 * np.log10(255.0 / rmse)
	return psnr


def main():
	# folder path
	q = 10
	folder = '../datas/Classic5'
	# folder = '../datas/LIVE1'

	folder_comp = '../datas/Classic5_q' + str(q)
	# folder_comp = '../datas/LIVE1_q' + str(q)

	# generate the file list
	filepath = os.listdir(folder)
	filepath.sort()
	
	im_input = tf.placeholder('float', [1, 1, None, None], name='im_input')

	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	with tf.device('/gpu:0'):
		with open('./graph_q%d.pb' % q, 'rb') as f: 
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			output = tf.import_graph_def(graph_def, input_map={'im_input:0': im_input}, return_elements=['output:0'])
	
	sum_psnr = 0
	# 29.547 (Classic5) 29.388 (LIVE1)
	for i in np.arange(0, len(filepath), 1):
		im = np.array(Image.open(os.path.join(folder, filepath[i])))
		im_n = np.array(Image.open(os.path.join(folder_comp, filepath[i][:-3] + 'jpg')))
		# (Image.fromarray(im)).show()
		
		im_n = im_n.astype(np.float32) / 255.0
		im_n = np.expand_dims(im_n, axis=0)
		im_n = np.expand_dims(im_n, axis=0)
		
		im_dn = sess.run(output, feed_dict={im_input: im_n})
		im_dn = np.squeeze(im_dn)
		
		im = shave(im, [1, 1])
		im_dn = shave(im_dn, [1, 1])
		
		im_dn[im_dn>1] = 1
		im_dn[im_dn<0] = 0

		psnr = compute_psnr(im, im_dn * 255)
		print('%d: %.2f dB' % (i + 1, psnr))
		sum_psnr += psnr
		
	avg_psnr = sum_psnr / len(filepath)
	print('Averaged PSNR: %.4f dB' % avg_psnr)


if __name__ == '__main__':
	main()
