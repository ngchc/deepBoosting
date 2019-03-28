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
	mean_folder = '../datas/Set15/mean'
	noise_folder = '../datas/Set15/noise'
	
	# generate the file list
	filepath = os.listdir(noise_folder)
	filepath.sort()
	
	im_input = tf.placeholder('float', [1, None, None, 3], name='im_input')

	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	sum_psnr = 0
	for i in np.arange(0, len(filepath), 1):
		if i == 0: gp = '5d-32.pb'
		elif i == 3: gp = 'd6-32.pb'
		elif i == 6: gp = 'd8-16.pb'
		elif i == 9: gp = 'd8-32.pb'
		elif i == 12: gp = 'd8-64.pb'
		
		if i % 3 == 0:
			with tf.device('/gpu:0'):
				with open('./graphs/' + gp, 'rb') as f: 
					graph_def = tf.GraphDef()
					graph_def.ParseFromString(f.read())
					output = tf.import_graph_def(graph_def, input_map={'im_input:0': im_input}, return_elements=['output:0'])
		
		im = np.array(Image.open(os.path.join(mean_folder, filepath[i][:-8] + 'mean.png')))
		im_n = np.array(Image.open(os.path.join(noise_folder, filepath[i])))
		
		im_n = im_n.astype(np.float32) / 255.0
		im_n = np.expand_dims(im_n, axis=0)
		
		im_dn = sess.run(output, feed_dict={im_input: im_n})
		im_dn = np.squeeze(im_dn)
		
		im_dn = np.minimum(im_dn, 1.0)
		im_dn = np.maximum(im_dn, 0.0)
		
		psnr = compute_psnr(im, (im_dn * 255).astype(np.uint8))
		print('%d: %.2f dB' % (i + 1, psnr))
		sum_psnr += psnr
		
		#Image.fromarray((im_dn * 255).astype(np.uint8)).save('./results-set15/' + filepath[i][:-9] + '.png')
	
	avg_psnr = sum_psnr / len(filepath)
	print('Averaged PSNR: %.4f dB' % avg_psnr)


if __name__ == '__main__':
	main()
