"""
Instantiate a solver for training
"""

import tensorflow as tf
from os import system, path
from solver import *


flags = tf.app.flags

# settings of the solver and hyper-parameters
flags.DEFINE_string('data_dir', './', 'path of the training dataset')
flags.DEFINE_string('models_dir', './models', 'trained model save path')
flags.DEFINE_string('log_name', 'ddfn.log', 'name of the log file')

# setting of the gpu devices
flags.DEFINE_integer('device_id', 0, 'assign the first id number')

# policy of the leanrning rate
flags.DEFINE_float('base_lr', 0.001, 'the basic (initial) learning rate')
flags.DEFINE_float('power', 1.5, 'power of the polynomial')
flags.DEFINE_float('end_lr', 0.0001, 'the minimal end learning rate')

# epoch and batch size
flags.DEFINE_integer('num_epoch', 800, 'num of training epoch')
flags.DEFINE_integer('batch_size', 64, 'batch size of the training dataset')

# regularization
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')

# resuming and finetune
flags.DEFINE_boolean('resume', False, 'whether to resume from the trained variables')
flags.DEFINE_boolean('finetune', False, 'whether to finetune from the trained variables')
flags.DEFINE_string('finetune_models_dir', './models', 'the path for searching the trained model')
flags.DEFINE_integer('iters', None, 'iteration of the trained variable')

conf = flags.FLAGS


def main(_):
	solver = Solver()
	solver.train(disp_freq=100, save_freq=900, summary_freq=100)


if __name__ == '__main__':
	tf.app.run()
