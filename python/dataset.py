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
    
    def __init__(self, scan_numbers, batch_size, augment_channels=False, augment_images=False):
        '''
            scan_numbers must be a list
        '''
        
        self.x_transformed = {} # start off with no transformation
        self.y_transformed = {}
        self.scan_numbers = scan_numbers
        
        # load data
        for scan_idx, scan_number in enumerate(self.scan_numbers):
            im_ref, im_us = get_dataset(scan_number)
        
            if (augment_channels == True):
                im_us = augment_channel_image(im_us)
            
            print('X shape: ', im_us.shape)
            print('y shape: ', im_ref.shape)
            
            self.x_transformed[scan_idx] = im_us
            self.y_transformed[scan_idx] = im_ref
        
                
        self.batch_size = batch_size
        self.epoch_number = 0
        self.slices_in_volume = im_ref.shape[0]     
        self.augment_images = augment_images
        
        
        print('augment_images: ', self.augment_images)

    def __len__(self):
        return len(self.scan_numbers) * int(np.ceil((self.slices_in_volume) / float(self.batch_size)))

    def __getitem__(self, idx):                
        
        scan_idx = int(np.floor(float(idx) * self.batch_size / self.slices_in_volume)) # generate a number between 0, number of scans

        idx_min = (idx * self.batch_size) - (self.slices_in_volume * scan_idx)  # generate a number between 0, slices_in_volume inclusive
        idx_max = ((idx + 1) * self.batch_size) - (self.slices_in_volume * scan_idx)
                
        batch_x = self.x_transformed[scan_idx][idx_min:idx_max, :, :, :]
        batch_y = self.y_transformed[scan_idx][idx_min:idx_max, :, :, :]
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        # augment dataset with transform
        self.epoch_number += 1
        
        if (self.augment_images == True):
            # alternating between horizontal flips and vertical flips covers all possible transformations
            transformation = self.epoch_number % 2 + 1

            for scan_idx, scan_number in enumerate(self.scan_numbers):
                self.x_transformed[scan_idx], self.y_transformed[scan_idx] = transform_image(self.x_transformed[scan_idx], self.y_transformed[scan_idx], transformation) # lazy transform lets us do it in-place
''' end MRImageSequence '''        
        
## example from https://keras.io/callbacks/
class LossHistory(tf.keras.callbacks.Callback):
    
    def __init__(self, test_data = None):
        self.test_data = test_data
        self.skip = 9 
    
    def on_train_begin(self, logs={}):
        self.train_losses_batch = []
        self.train_losses_epoch = []
        self.test_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses_batch.append(logs.get('loss'))
        
    def on_epoch_end(self, epochs, logs={}):
        self.train_losses_epoch.append(logs.get('loss'))
        
        if (epochs % self.skip == 0 and self.test_data != None):
            x, y = self.test_data
            loss, _ = self.model.evaluate(x, y, verbose=0)
            self.test_losses.append(loss)
''' end LossHistory '''                

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