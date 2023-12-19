from Layer_ITNE import Layer_ITNE
import tensorflow as tf

class Virtual_ITNE(tf.keras.layers.Layer, Layer_ITNE):
    def __init__(self, input_shape=None):
        super().__init__()
        Layer_ITNE.__init__(self)

    def call(self, inputs, **kwargs):
        return inputs

    def update_shape(self, input_shape):
        self.shape = input_shape

    def eyes(self):
        if isinstance(self.shape, int):
            eye = tf.eye(self.shape)
        else:
            shape = list(self.shape)
            size = tf.reduce_prod(shape)
            eye = tf.eye(size)
            eye = tf.reshape(eye, shape+shape)
        return tf.stack([eye, -eye])