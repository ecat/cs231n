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
    
	im_ref = im_ref[:, :, :, np.newaxis] # put into format (samples, rows, cols, channels)


	return (im_ref, im_us)