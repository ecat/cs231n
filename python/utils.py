import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

from collections import Sequence
import math

def gen_interm_fun(inputs, layer):

    fun = tf.keras.backend.function(inputs=[inputs], outputs=[layer.output])

    def _forward_pass(images):
        assert len(images.shape) == 3 or len(images.shape) == 4

        if len(images.shape) == 3:
            l = list(images.shape)
            l.insert(0, 1)
            images = tf.reshape(images, tuple(l))

        out = fun([images])
        # .eval(session=tf.keras.backend.get_session())

        return out[0]

    return _forward_pass


def plot_convs(layer, channels=None, filters=None):
    '''
    Plots heatmaps for kernels

    channels: An iterable of channels to plot. If None, plots all of them
    filters: An iterable of filters to plot. If None, plots all of them
    '''
    sublayers = layer.get_weights()
    convs = sublayers[0]
    biases = sublayers[1]

    if channels == None:
        channels = range(convs.shape[2])
    if filters == None:
        filters = range(convs.shape[-1])

    if not isinstance(channels, Sequence):
        channels = [channels]
    if not isinstance(filters, Sequence):
        filters = [filters]

    cols = 4 # seems like a good number
    rows = int(math.ceil((len(channels) * len(filters)) / 4.0))

    fig, axes = plt.subplots(nrows=rows, ncols=cols)

    idx = 0

    cmin = np.amin(convs)
    cmax = np.amax(convs)

    for i in channels:
        for j in filters:
            idx += 1
            ax = plt.subplot(rows, cols, idx)
            ax.set_title('filter={} channel={}'.format(j, i))
            im = ax.imshow(convs[:,:,i,j], cmap=plt.cm.coolwarm, vmin=cmin, vmax=cmax)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.tight_layout()

    plt.rcParams['figure.figsize'] = [2*cols, 2*rows]

    # Make an axis for the colorbar on the right side
    cax = fig.add_axes([1.2, 0.1, 0.08, 0.8])
    cbar = fig.colorbar(im, cax=cax)

    return plt
