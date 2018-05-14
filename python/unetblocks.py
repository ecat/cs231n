import tensorflow as tf

F=16
kernel_size = (3, 3)
strides = (1, 1)

def res_block(x, resblocks=[]):
    if len(resblocks) > 0:
        resblock = resblocks[0]
        resblocks = resblocks[1:]
        c1 = tf.keras.layers.Conv2D(filters=F, kernel_size=kernel_size, padding='same', strides=strides, activation='relu')(x)
        m1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(c1)
        rb1 = resblock(m1, resblocks)
        up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(rb1)
        concatenated_input = tf.keras.layers.concatenate([up1, x], 3)
        y = tf.keras.layers.Conv2D(filters=F, kernel_size=kernel_size, padding='same', strides=strides, activation='relu')(concatenated_input)
    else:
        # bottom of the bowl
        y = tf.keras.layers.Conv2D(filters=F, kernel_size=kernel_size, padding='same', strides=strides)(x)

    return y
