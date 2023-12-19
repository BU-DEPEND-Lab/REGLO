from Layer_ITNE import Layer_ITNE
import tensorflow as tf

class Normalization_ITNE(tf.keras.layers.experimental.preprocessing.Normalization, Layer_ITNE):
    def __init__(self, axis=-1, dtype=None, **kwargs):
        super().__init__(axis, dtype, **kwargs)
        Layer_ITNE.__init__(self)

    def backward(self, last_dA, inputs_bounds):
        dA = tf.math.multiply(last_dA, 1/tf.sqrt(self.variance))
        bias = 0
        return [dA], bias

    def update_shape(self, input_shape):
        self.my_input_shape = input_shape

    def eyes(self):
        if isinstance(self.my_input_shape, int):
            eye = tf.eye(self.my_input_shape)
        else:
            shape = list(self.my_input_shape)
            size = tf.reduce_prod(shape)
            eye = tf.eye(size)
            eye = tf.reshape(eye, shape+shape)
        return tf.stack([eye, -eye])