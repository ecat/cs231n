import tensorflow as tf

def _proc_unmaxpool(*args, **kwargs):
    if 'pool_size' in kwargs and 'size' not in kwargs:
        kwargs['size'] = kwargs['pool_size']
    for n in {'pool_size', 'strides', 'padding'}:
        if n in kwargs:
            del kwargs[n]
    return tf.keras.layers.UpSampling2D(*args, **kwargs)

layermap = {'conv': tf.keras.layers.Conv2D,
            'maxpool': tf.keras.layers.MaxPool2D,
            'batchnorm': tf.layers.BatchNormalization,
            'flatten': tf.keras.layers.Flatten,
            'fc': tf.keras.layers.Dense,
            'rev': lambda : 'rev',
            'unconv': tf.keras.layers.Conv2D,
            'unmaxpool': _proc_unmaxpool
            }

def gen_layer(name):
    if name not in layermap and 'un{}'.format(name) not in layermap:
        raise RuntimeError("Unknown layer name {}".format(name))

    class Layer:
        def __init__(self, *args, **kwargs):
            self.__name__ = name.replace('un','')
            self.args = args
            self.kwargs = kwargs

            if name in layermap:
                self._undo = False
            else:
                # This is an undo layer, e.g. unconv
                self._undo = True

            self.fun = layermap[name]

        @property
        def name(self):
            return self.__name__

        @property
        def undo(self):
            return self._undo

        def flip(self):
            if self._undo:
                return gen_layer(self.name)(*self.args, **self.kwargs)
            else:
                return gen_layer('un' + self.name)(*self.args, **self.kwargs)

        def __str__(self):
            val = 'un' + self.name if self._undo else self.name
            val += self.args.__repr__() + ',' + self.kwargs.__repr__()
            return val

        def __repr__(self):
            return self.__str__()

        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.args == other.args and self.kwargs == other.kwargs and self.fun == other.fun

        def __hash__(self):
            return tuple(self.name, self.args, self.kwargs, self._undo).__hash__()

    return Layer

# Shortcuts
shortcuts = lambda : 0
for k, v in layermap.items():
    setattr(shortcuts, k, gen_layer(k))
