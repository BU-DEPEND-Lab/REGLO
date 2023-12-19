from Layer_ITNE import Layer_ITNE
import tensorflow as tf

class Conv2d_ITNE(tf.keras.layers.Conv2D, Layer_ITNE):
    def __init__(self, filters, kernel_size, strides=1, data_format=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(filters, kernel_size, strides, 'valid', data_format, (1, 1), None, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        Layer_ITNE.__init__(self)
        self.my_strides = strides

            
    def backward(self, last_dA, inputs_bounds):
        # dA = tf.matmul(last_dA, self.kernel, transpose_b=True)
        dA_shape = list(last_dA.shape[:-3]) + list(self.my_input_shape)
        batch_size = tf.reduce_prod(last_dA.shape[:-3])
        last_dA = tf.reshape(last_dA,[batch_size] + last_dA.shape[-3:])
        output_shape = [batch_size] + self.my_input_shape
        dA = tf.reshape(tf.nn.conv2d_transpose(last_dA, self.kernel, output_shape, self.my_strides, padding="VALID"), dA_shape)
        bias = 0
        return [dA], bias

    def update_shapes(self, input_shape, output_shape):
        self.my_input_shape = input_shape
        self.my_output_shape = output_shape

    def eyes(self):
        size = tf.reduce_prod(self.my_output_shape)
        eye = tf.eye(size)
        shape = list(self.my_output_shape)
        eye = tf.reshape(eye, shape+shape)
        return tf.stack([eye, -eye])