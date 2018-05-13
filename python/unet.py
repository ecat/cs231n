from utils import shortcuts
import tensorflow as tf
from collections import Sequence

class UNet:
    def __init__(self, input_shape, *args):
        if len(args) == 1 and isinstance(args[0], Sequence):
            # handle single sequence
            args = tuple(args[0])

        self.layers = list(args)
        if self.layers[-1] == shortcuts.rev():
            self.layers = self.layers[:-1]
            self.layers += [l.flip() for l in reversed(self.layers)]
        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self._model = None
        self._lvals = {}
        self._gen_graph()
        self._model = tf.keras.models.Model(inputs=self.inputs, outputs=self.res)

    def _gen_graph(self):
        x = self.inputs
        lvl = 0
        shapes = []
        self._lvals[(lvl, False)] = x
        for l in self.layers:
            if l.name.replace('un', '') == shortcuts.maxpool.name:
                if l.undo:
                    lvl -= 1
                else:
                    lvl += 1

            x = l.fun(*l.args, **l.kwargs)(x)
            shapes.append((l.name, x.shape))

            if l.undo:
                # need to crop and concatenate
                cropped_prev = self._match_size(self._lvals[(lvl, False)], x)
                x = tf.concat([cropped_prev, x], 3)

            self._lvals[(lvl, l.undo)] = x

        # reshape output
        '''
        filtered_shapes = list(filter(lambda t: t[0] not in {'flatten', 'fc'}, shapes))
        target_shape = list(filtered_shapes[-1][1][1:3])
        target_shape.append(1) # nani??
        self.res = tf.keras.layers.Reshape(tuple(target_shape))(x)
        '''
        self.res = x
        
        return self.res

    def _match_size(self, layer1, layer2):
        '''
        Crops layer1 to match layer2
        '''
        offsets = [0, (layer1.shape[1] - layer2.shape[1]) // 2, (layer1.shape[2] - layer2.shape[2]) // 2, 0]
        sizes = [-1, layer2.shape[1], layer2.shape[2], -1]
        return tf.slice(layer1, offsets, sizes)

    def train(self, X_train, y_train, optimizer='adam', loss='mean_squared_error', metrics=['mse']):
        
        ## example from https://keras.io/callbacks/
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
                
        history = LossHistory()
        
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._model.fit(x=X_train, y=y_train, callbacks=[history])
        
        return history

    def eval(self, inputs):
        return self.model(inputs)
    
    def predict(self, inputs):
        return self.model.predict(inputs)

    def __call__(self, inputs):
        '''
        Evaluates the model
        '''
        return self.model(inputs)

    @property
    def model(self):
        return self._model
