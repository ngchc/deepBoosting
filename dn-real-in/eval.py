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
	folder = '../datas/Set60/ISO6400'

	# generate the file list
	filepath = os.listdir(folder)
	filepath.sort()
	
	im_input = tf.placeholder('float', [1, None, None, 3], name='im_input')

	# create a session for running operations in the graph
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	with tf.device('/gpu:0'):
		with open('./graph.pb', 'rb') as f: 
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			output = tf.import_graph_def(graph_def, input_map={'im_input:0': im_input}, return_elements=['output:0'])
	
	record_psnr = []
	for i in np.arange(1, 20+1, 1):
		for p in np.arange(1, 3+1, 1):
			psnrs = []
			im = np.array(Image.open(os.path.join(folder, '%03d/%03dMP%d.PNG' % (i, i, p))))
			#Image.fromarray(im).show()
			for g in np.arange(1, 10+1, 1):
				im_n = np.array(Image.open(os.path.join(folder, '%03d/%03dN%02dP%d.PNG' % (i, i, g, p))))
				#Image.fromarray(im_n).show()

				im_n = im_n.astype(np.float32) / 255.0
				im_n = np.expand_dims(im_n, axis=0)

				im_dn = sess.run(output, feed_dict={im_input: im_n})
				im_dn = np.squeeze(im_dn) * 255.0

				im_dn = np.maximum(im_dn, 0)
				im_dn = np.minimum(im_dn, 255)
				#Image.fromarray(np.asarray(im_dn, dtype=np.uint8)).show()

				psnr = compute_psnr(im, np.asarray(im_dn, dtype=np.uint8))
				print('i%03d p%d g%02d: %.2f dB' % (i, p, g, psnr))
				psnrs.append(psnr)

			record_psnr.append(psnrs)

	print('%.2f+-%.3f dB' % (np.mean(record_psnr), np.mean(np.std(record_psnr, 1))))


if __name__ == '__main__':
	main()
