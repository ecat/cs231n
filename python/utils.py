import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

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


def plot_convs(layer, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    filters = layer.get_weights()

    convs = filters[0]
    biases = filters[1]
    idx = 0

    cmin = np.amin(convs)
    cmax = np.amax(convs)

    for i in range(rows):
        for j in range(cols):
            idx += 1
            ax = plt.subplot(rows, cols, idx)
            im = ax.imshow(convs[:,:,i,j], cmap=plt.cm.coolwarm, vmin=cmin, vmax=cmax)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.tight_layout()

    plt.rcParams['figure.figsize'] = [cols, rows]

    # Make an axis for the colorbar on the right side
    cax = fig.add_axes([1.2, 0.1, 0.08, 0.8])
    cbar = fig.colorbar(im, cax=cax)

    return plt
