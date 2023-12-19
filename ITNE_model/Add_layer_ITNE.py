from Layer_ITNE import Layer_ITNE
import tensorflow as tf

class Add_ITNE(tf.keras.layers.Add, Layer_ITNE):
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        Layer_ITNE.__init__(self)
        self.shape = input_shape
        
    def update_shape(self, input_shape):
        self.shape = input_shape

    def backward(self, last_dA, inputs_bounds):
        bias = 0
        return [last_dA, last_dA], bias

    def eyes(self):
        size = tf.reduce_prod(self.shape)
        eye = tf.eye(size)
        shape = list(self.shape)
        eye = tf.reshape(eye, shape+shape)
        return tf.stack([eye, -eye])