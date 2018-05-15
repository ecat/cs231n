import tensorflow as tf

def res_block(Conv1, Conv2=None):
    def _res_block_helper(x, resblocks=[]):
        if len(resblocks) > 0:
            resblock = resblocks[0]
            resblocks = resblocks[1:]
            c1 = Conv1(x)
            m1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(c1)
            rb1 = resblock(m1, resblocks)
            up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(rb1)
            concatenated_input = tf.keras.layers.concatenate([up1, x], 3)
            y = Conv2(concatenated_input)
        else:
            # bottom of the bowl
            if Conv2 is not None:
                print("Warning: Not using second convolution at the bottom of the U.")
            y = Conv1(x)

        return y
    return _res_block_helper

def gen_conv(filters, kernel_size, strides=(1, 1), padding='same', activation='relu', **kwargs):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, **kwargs)


def gen_conv_bn_relu(**kwargs):               
    
    layers = [
        gen_conv(**kwargs),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu')
    ]
    
    def gen_layers(x):
        for layer in layers:
            x = layer(x)            
        return x
    
    return gen_layers
