import tensorflow as tf

def res_block(Conv1, Conv2=None, use_pool=True):
    def _res_block_helper(x, resblocks=[]):
        if len(resblocks) > 0:
            # init first block
            resblock = resblocks[0]
            resblocks = resblocks[1:]
            c1 = Conv1(x)
            
            if(use_pool):
                m1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(c1)
            else:
                m1 = c1
            
            # init outputs of first block
            rb1 = resblock(m1, resblocks)
            
            if(use_pool):
                up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(rb1)
            else:
                up1 = rb1
            
            # combine output of first block, incoming resnet outputs
            concatenated_input = tf.keras.layers.concatenate([up1, x], 3)
            
            # init second block
            y = Conv2(concatenated_input)
        else:
            # bottom of the bowl
            if Conv2 is not None:
                print("Warning: Not using second convolution at the bottom of the U.")
            y = Conv1(x)

        return y
    return _res_block_helper

def gen_conv(filters, kernel_size, strides=(1, 1), padding='same', **kwargs):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='linear', **kwargs)

def gen_conv_relu(N=1, **kwargs):
    
    def gen_layers(x): 
        for ii in range(N):
            layers = [ # need to define layers within function because you can't reuse them between N
                gen_conv(**kwargs),
                tf.keras.layers.Activation('relu')
            ]
            for layer in layers:
                x = layer(x)
        return x
    
    return gen_layers


def gen_conv_bn_relu(N=1, **kwargs):               
    
    def gen_layers(x):
        for ii in range(N):
            layers = [
                gen_conv(**kwargs),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu')
            ]                
            for layer in layers:
                x = layer(x)            
        return x
    
    return gen_layers
