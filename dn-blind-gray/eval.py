import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matlab.engine


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
	# folder = '../datas/Set12'
	folder = '../datas/BSD68'

	# generate the file list
	filepath = os.listdir(folder)
	filepath.sort()
	
	# 5.0 ~ 50.0
	sigma = 45.0
	im_input = tf.placeholder('float', [1, 1, None, None], name='im_input')

	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	with tf.device('/gpu:0'):
		with open('./graph.pb', 'rb') as f: 
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			output = tf.import_graph_def(graph_def, input_map={'im_input:0': im_input}, return_elements=['output:0'])
	
	sum_psnr = 0
	for i in np.arange(0, len(filepath), 1):
		im = np.array(Image.open(os.path.join(folder, filepath[i])))
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
		print('%d: %.2f dB' % (i + 1, psnr))
		sum_psnr += psnr
	
	avg_psnr = sum_psnr / len(filepath)
	print('Averaged PSNR: %.4f dB' % avg_psnr)


if __name__ == '__main__':
	eng = matlab.engine.start_matlab()
	main()
	eng.quit()
