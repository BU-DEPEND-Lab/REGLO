from Layer_ITNE import Layer_ITNE
import tensorflow as tf

class Flatten_ITNE(tf.keras.layers.Flatten, Layer_ITNE):
    def __init__(self, input_shape=None, data_format=None, **kwargs):
        super().__init__(data_format, **kwargs)
        Layer_ITNE.__init__(self)
        self.my_input_shape = input_shape

    def update_shape(self, input_shape):
        self.my_input_shape = input_shape

    def backward(self, last_dA, inputs_bounds):
        dA_shape = list(last_dA.shape[:-1]) + list(self.my_input_shape)
        dA = tf.reshape(last_dA, dA_shape)
        bias = 0
        return [dA], bias

    def eyes(self):
        size = tf.reduce_prod(self.my_input_shape)
        eye = tf.eye(size)
        return tf.stack([eye, -eye])
