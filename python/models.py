import tensorflow as tf
from unetblocks import res_block, get_gen_conv_fn

def get_very_small_unet(inputs, use_pool, use_bn=False):
    '''
    inputs is a keras layer that can be created with    
    inputs = tf.keras.layers.Input(shape=input_shape)
    '''
    
    gen_fn = get_gen_conv_fn(use_bn)
    
    print('get_very_small_unet')
    print('use_pool:', use_pool)
    print('gen_fn: ', gen_fn.__name__)
    
    # very small unet
    res_out = res_block(gen_fn(N=2, **f16), gen_fn(N=2, **f16), use_pool=False)(inputs, 
                                                      resblocks=[res_block(gen_fn(N=3, **f32))])
    out = tf.keras.layers.Dense(1)(res_out)
        
    return out

def get_small_unet(inputs, use_bn=False):
    
    gen_fn = get_gen_conv_fn(use_bn)
    
    print('get_small_unet')
    print('use_pool: ', True)
    print('gen_fn: ', gen_fn.__name__)
    
    # small unet    
    res_out = res_block(gen_fn(N=2, **f16), gen_fn(N=2, **f16))(inputs, 
                                                          resblocks=[res_block(gen_fn(N=3, **f32), gen_fn(N=3, **f32)),
                                                                     res_block(gen_fn(N=3, **f64), gen_fn(N=3, **f64)),
                                                                             res_block(gen_fn(N=3, **f128))])
    out = tf.keras.layers.Dense(1)(res_out)
    
    return out

def get_kaist_unet(inputs, use_bn=False):
    
    gen_fn = get_gen_conv_fn(use_bn)
    
    print('get_kaist_unet')
    print('gen_fn: ', gen_fn.__name__)

    # big unet
    res_out = res_block(gen_fn(N=2, **f32), gen_fn(N=2, **f32))(inputs, 
                                                          resblocks=[res_block(gen_fn(N=2, **f64), gen_fn(N=2, **f64)),
                                                                     res_block(gen_fn(N=2, **f128), gen_fn(N=2, **f128)),
                                                                     res_block(gen_fn(N=2, **f256), gen_fn(N=2, **f256)),
                                                                             res_block(gen_fn(N=2, **f512))])
    out = tf.keras.layers.Dense(1)(res_out)
    
    return out


gen_conv_params = lambda num_filters : {'filters': num_filters, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same'}
f16 = gen_conv_params(16)
f32 = gen_conv_params(32)
f64 = gen_conv_params(64)
f128 = gen_conv_params(128)
f256 = gen_conv_params(256)
f512 = gen_conv_params(512)
f1024 = gen_conv_params(1024)