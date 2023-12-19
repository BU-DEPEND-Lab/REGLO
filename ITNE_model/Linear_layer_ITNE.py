from Layer_ITNE import Layer_ITNE
import tensorflow as tf

class Linear_ITNE(tf.keras.layers.Dense, Layer_ITNE):
    def __init__(self, units, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(units, None, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        Layer_ITNE.__init__(self)

    def backward(self, last_dA, inputs_bounds):
        dA = tf.matmul(last_dA, self.kernel, transpose_b=True)
        bias = 0
        return [dA], bias

    def eyes(self):
        eye = tf.eye(self.units)
        return tf.stack([eye, -eye])