"""
Data generator for train400
"""

import tensorflow as tf
import numpy as np


# the actual number of datasets WITH regard to the build-in augmentation
num_examples_per_epoch_for_train = 115200


def gaussian_noise(sigma):
	noise = tf.random_normal([1, 50, 50], mean=0.0, stddev=sigma, dtype=tf.float32)
	return noise


class Train400_Data(object):
	def __init__(self, filename, num_epoch, batch_size, sigma, shuffle=True, augmentation=False, scope='train400_data'):
		with tf.name_scope(scope) as scope:
			# create a queue that produces the filenames to read
			filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch, capacity=100)
			reader = tf.TFRecordReader()
			
			_, serialized_example = reader.read(filename_queue)
			features = tf.parse_single_example(serialized_example, features = {
				'label': tf.FixedLenFeature([], tf.string)
			})
			
			# convert from a string to a vector of uint8
			label = tf.decode_raw(features['label'], tf.uint8)
			label = tf.reshape(label, [1, 50, 50])
			label = tf.cast(label, tf.float32) / 255.0
			
			# generate noise and input data
			noise = gaussian_noise(sigma / 255.0)
			data = label + noise
			
			# data augmentation (save memory but degrade speed)
			if shuffle and augmentation:
				augs = tf.constant([0, 1, 2, 3, 4, 5, 6, 7], dtype=tf.int32)
				
				with tf.name_scope('augmentation'):
					# callable lambda functions
					def aug0(data, label):
						return data, label
					def aug1(data, label):
						data = tf.image.rot90(data, k=1)
						label = tf.image.rot90(label, k=1)
						return data, label
					def aug2(data, label):
						data = tf.image.rot90(data, k=2)
						label = tf.image.rot90(label, k=2)
						return data, label
					def aug3(data, label):
						data = tf.image.rot90(data, k=3)
						label = tf.image.rot90(label, k=3)
						return data, label
					def aug4(data, label):
						data = tf.image.flip_left_right(data)
						label = tf.image.flip_left_right(label)
						return data, label
					def aug5(data, label):
						data = tf.image.flip_up_down(data)
						label = tf.image.flip_up_down(label)
						return data, label
					def aug6(data, label):
						data = tf.image.flip_left_right(tf.image.rot90(data, k=1))
						label = tf.image.flip_left_right(tf.image.rot90(label, k=1))
						return data, label
					def aug7(data, label):
						data = tf.image.flip_up_down(tf.image.rot90(data, k=1))
						label = tf.image.flip_up_down(tf.image.rot90(label, k=1))
						return data, label
					
					k = tf.random_uniform([], 0, 7, tf.int32)
					data, label = tf.case(pred_fn_pairs=
					                      [(tf.equal(k, augs[1]), (lambda d,l: lambda: aug1(d,l))(data, label)),
					                       (tf.equal(k, augs[2]), (lambda d,l: lambda: aug2(d,l))(data, label)),
					                       (tf.equal(k, augs[3]), (lambda d,l: lambda: aug3(d,l))(data, label)),
					                       (tf.equal(k, augs[4]), (lambda d,l: lambda: aug4(d,l))(data, label)),
					                       (tf.equal(k, augs[5]), (lambda d,l: lambda: aug5(d,l))(data, label)),
					                       (tf.equal(k, augs[6]), (lambda d,l: lambda: aug6(d,l))(data, label)),
					                       (tf.equal(k, augs[7]), (lambda d,l: lambda: aug7(d,l))(data, label))],
						                  default=(lambda d,l: lambda: aug0(d,l))(data, label),
						                  exclusive=True)
					
					data = tf.reshape(data, [14, 14, 1])
					label = tf.reshape(label, [28, 28, 1])
			
			# ensure that the random shuffling has good mixing properties
			min_fraction_of_examples_in_queue = 0.4
			min_queue_examples = int(num_examples_per_epoch_for_train * min_fraction_of_examples_in_queue)
			
			# generate a batch of images and labels by building up a queue of examples
			num_preprocess_threads = 8
			if shuffle:
				self.datas, self.labels = tf.train.shuffle_batch([data, label], batch_size=batch_size,
					                                             num_threads=num_preprocess_threads,
					                                             capacity=min_queue_examples + 3 * batch_size,
					                                             min_after_dequeue=min_queue_examples,
				                                                 allow_smaller_final_batch=False)
			else:
				self.datas, self.labels = tf.train.batch([data, label], batch_size=batch_size,
					                                     num_threads=num_preprocess_threads,
					                                     capacity=min_queue_examples + 3 * batch_size)
