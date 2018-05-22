import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cfl as cfl # functions for reading raw mr files

from enum import IntEnum

class Transform(IntEnum):
    NONE = 0
    FLIP_HOR = 1
    FLIP_VER = 2
    ROT_180 = 3
    
class MRImageSequence(tf.keras.utils.Sequence):
    
    def __init__(self, scan_number, batch_size, augment_channels=False):
        
        im_ref, im_us = get_dataset(scan_number)
        
        if (augment_channels == True):
            im_us = augment_channel_image(im_us)
        
        self.x_ref = im_us
        self.y_ref = im_ref        
        
        self.x_transformed = im_us.copy() # start off with no transformation
        self.y_transformed = im_ref.copy()
        self.batch_size = batch_size
        self.epoch_number = 0
        
        print('X size: ', self.x_ref.shape)
        print('y size: ', self.y_ref.shape)

    def __len__(self):
        return int(np.ceil((self.x_ref.shape[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        idx_min = idx * self.batch_size
        idx_max = (idx + 1) * self.batch_size
        
        batch_x = self.x_transformed[idx_min:idx_max, :, :, :]
        batch_y = self.y_transformed[idx_min:idx_max, :, :, :]

        return batch_x, batch_y
    
    def on_epoch_end(self):
        
        # augment dataset with transform
        self.epoch_number += 1
        
        self.x_transformed, self.y_transformed = transform_image(self.x_transformed, self.y_transformed, self.epoch_number % 4) # lazy transform, do a copy, could technically do it inplace
        
        
    

def get_dataset(scan_number):
    '''
    Loads an arc undersampled dataset that is real
    im_ref is a coil comabined image, is size (N, H, W, 1)
    im_us is 2x undersampled real images, is size (N, H, W, C)
    '''
    print('loading scan ', scan_number)
    
    data_dir = 'data/'
    filename_ref = 'scan' + str(scan_number) + '_real_ref'
    filename_us = 'scan' + str(scan_number) + '_real_us'
    
    path_ref = data_dir + filename_ref
    path_us = data_dir + filename_us
    
    im_ref = np.real(cfl.readcfl(path_ref)).astype(np.float32)
    im_us = np.real(cfl.readcfl(path_us)).astype(np.float32)
    
    im_ref = im_ref[:, :, :, np.newaxis] # put into format (samples, rows, cols, channels)

    return (im_ref, im_us)

def get_dataset2(scan_number, do_real=True):
    '''
    Loads ksp data and applies an arc mask, while also converting to magnitude data if required
    im_ref is not a combined coil image, is size (N, H, W, C)
    im_us is size (N, H, W, C)
    '''
    data_dir = 'data2/'
    
    filename_ref = 'scan' + str(scan_number) + '_ref'
    filename_mask = 'scan' + str(scan_number) + '_mask'
    
    path_ref = data_dir + filename_ref
    path_mask = data_dir + filename_mask
    
    ksp_ref = cfl.readcfl(path_ref)
    
    if(do_real):
        ksp_ref = fft3c(np.abs(ifft3c(ksp_ref)))
    
    mask = np.real(cfl.readcfl(path_mask)).astype(np.float32)
    mask = mask[:, :, :, np.newaxis]
    
    # apply mask
    ksp_us = ksp_ref * mask
    
    im_ref = ifft3c(ksp_ref)
    im_us = ifft3c(ksp_us)
    
    if(do_real):
        im_ref = np.real(im_ref).astype(np.float32)
        im_us = np.real(im_us).astype(np.float32)
    
    return (im_ref, im_us, mask)


def transform_image(im_ref, im_us, which_transform):
    
    if which_transform == Transform.NONE: # do nothing
        im_ref_tf = im_ref
        im_us_tf = im_us
    elif which_transform == Transform.FLIP_HOR:
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
                     
def sos(im, axis):
    return np.sqrt(np.sum(np.power(im, 2), axis=axis))                     
                     
def fft3c(x):
    ax = [0, 1, 2]
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(x, axes=ax), axes=ax), axes=ax)

def ifft3c(x):
    ax = [0, 1, 2]
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax), axes=ax)