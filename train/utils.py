"""
Various utilities for evaluation
"""

import tensorflow as tf
import numpy as np


def softmax(x):
	"""
	Compute softmax values for each sets of scores in x
	"""
	e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
	return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference


def compute_psnr(im1, im2):
	"""
	Args:
	  im1/im2: [0, 255]

	Returns:
	  psnr: Peek Signal-to-Noise Ratio
	"""	
	if im1.shape != im2.shape:
		raise Exception('the shapes of two images are not equal')

	rmse = np.sqrt(((np.asfarray(im1) - np.asfarray(im2)) ** 2).mean())
	psnr = 20 * np.log10(255.0 / rmse)

	return psnr


def compute_psnr_from_mse(mse):
	rmse = np.sqrt(np.asarray(mse))
	psnr = 20 * np.log10(255.0 / rmse)
	return psnr


def img_stretch(img):
	"""
	Stretch (Normalize) the image to [0, 1]
	"""
	img = img.astype(float)
	img -= np.min(img)
	img /= np.max(img)+1e-12
	return img


def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0, stretch=False):
	"""
	Tile images in a grid
	Note: If tile_shape is provided only as many images as 
	      specified in tile_shape will be included in the output.
	"""

	# Prepare images
	if stretch:
		imgs = img_stretch(imgs)
	imgs = np.array(imgs)
	
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions.')
	n_imgs = imgs.shape[0]

	# Grid shape
	img_shape = np.array(imgs.shape[1:3])
	if tile_shape is None:
		img_aspect_ratio = img_shape[1] / float(img_shape[0])
		aspect_ratio *= img_aspect_ratio
		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
		grid_shape = np.array((tile_height, tile_width))
	else:
		assert len(tile_shape) == 2
		grid_shape = np.array(tile_shape)

	# Tile image shape
	tile_img_shape = np.array(imgs.shape[1:])
	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

	# Assemble tile image
	tile_img = np.empty(tile_img_shape)
	tile_img[:] = border_color
	for i in range(grid_shape[0]):
		for j in range(grid_shape[1]):
			img_idx = j + i*grid_shape[1]
			if img_idx >= n_imgs:
				# No more images - stop filling out the grid.
				break
			img = imgs[img_idx]
			yoff = (img_shape[0] + border) * i
			xoff = (img_shape[1] + border) * j
			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

	return tile_img


"""to be test!"""
def conv_filter_tile(filters):
	# pre-transpose HWCN to NCHW
	filters = np.transpose(filters, (3, 2, 0, 1))
	
	n_filters, n_channels, height, width = filters.shape
	tile_shape = None
	if n_channels == 3:
		# Interpret 3 color channels as RGB
		filters = np.transpose(filters, (0, 2, 3, 1))
	else:
		# Organize tile such that each row corresponds to a filter and the
		# columns are the filter channels
		tile_shape = (n_channels, n_filters)
		filters = np.transpose(filters, (1, 0, 2, 3))
		filters = np.resize(filters, (n_filters*n_channels, height, width))
	filters = img_stretch(filters)
	return img_tile(filters, tile_shape=tile_shape)
