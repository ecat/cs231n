import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cfl as cfl # functions for reading raw mr files

from enum import Enum

class Transform(Enum):
    FLIP_HOR = 1
    FLIP_VER = 2
    ROT_180 = 3

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


def transform_image(im_ref, im_us, which_transform):
    
    if which_transform == Transform.FLIP_HOR:
        im_ref_tf = np.flip(im_ref, axis=2) # flip along undersampling dimension
        im_us_tf = np.flip(im_us, axis=2)
    elif which_transform == Transform.FLIP_VER:
        im_ref_tf = np.flip(im_ref, axis=1) # flip along vertical
        im_us_tf = np.flip(im_us, axis=1)
    elif which_transform == Transform.ROT_180:
        im_ref_tf = np.flip(np.flip(im_ref, axis=2), axis=1) # rotate image 180 degrees using two flips
        im_us_tf = np.flip(np.flip(im_us, axis=2), axis=1)
    else:
        raise ValueError('Invalid transform value, use enum.')
        
    return (im_ref_tf, im_us_tf)
    
def augment_channel_image(im_us):
    N, H, W, C = im_us.shape
    
    im_roll = np.roll(im_us, W//2, axis=2)
    return np.concatenate((im_us, im_roll), axis=3)
    