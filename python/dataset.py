import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cfl as cfl # functions for reading raw mr files

def get_dataset(scan_number):
	data_dir = 'data/'
	filename_ref = 'scan' + str(scan_number) + '_real_ref'
	filename_us = 'scan' + str(scan_number) + '_real_us'
    
	path_ref = data_dir + filename_ref
	path_us = data_dir + filename_us

	im_ref = np.real(cfl.readcfl(path_ref)).astype(np.float32)
	im_us = np.real(cfl.readcfl(path_us)).astype(np.float32)

	# https://www.tensorflow.org/programmers_guide/datasets

	im_us_placeholder = tf.placeholder(im_us.dtype, im_us.shape)
	im_ref_placeholder = tf.placeholder(im_ref.dtype, im_ref.shape)

	dataset = tf.data.Dataset.from_tensor_slices((im_us_placeholder, im_ref_placeholder))

	return (dataset, im_ref, im_us)