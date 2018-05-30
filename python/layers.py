import tensorflow as tf

# Credit to JulesGM -- https://github.com/keras-team/keras/issues/3878
class LayerNorm1D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNorm1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=tf.keras.initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True,)

        super(LayerNorm1D, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNorm2D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNorm2D, self).__init__(**kwargs)

    def build(self, input_shape):
        ishape = input_shape.as_list()
        ishape = ishape[1:3]
        ishape.append(1)
        ishape = tf.TensorShape(ishape)
        self.gamma = self.add_weight(name='gamma',
                                     shape=ishape,
                                     initializer=tf.keras.initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=ishape,
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True,)

        super(LayerNorm2D, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
