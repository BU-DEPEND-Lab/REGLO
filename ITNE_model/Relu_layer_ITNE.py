from Layer_ITNE import Layer_ITNE
import tensorflow as tf


class Relu_ITNE(tf.keras.layers.ReLU, Layer_ITNE):
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(None, 0, 0, **kwargs)
        Layer_ITNE.__init__(self)
        self.need_bounds = True
        self.shape = input_shape

    def update_shape(self, input_shape):
        self.shape = input_shape
    
    def backward(self, last_dA, inputs_bounds: list):
        dy_lb = inputs_bounds[0][0]
        dy_ub = -inputs_bounds[0][1]
        dA = last_dA
        bias = 0
        dy_lb = tf.math.minimum(dy_lb, 0)
        dy_ub = tf.math.maximum(dy_ub, 0)
        dy_ub = tf.math.maximum(dy_ub, dy_lb + 1e-8)

        dy_diff = dy_ub - dy_lb

        dy_upper_d = tf.math.divide(dy_ub, dy_diff)
        dy_upper_b = tf.math.multiply(-dy_lb, dy_upper_d)
        dy_lower_d = tf.math.divide(-dy_lb, dy_diff)
        dy_lower_b = tf.math.multiply(-dy_ub, dy_lower_d)

        def _bound_oneside(A, d_pos, d_neg, b_pos, b_neg, dim):
            # multiply according to sign of A (we use fused operation to save memory)
            # neg_A = last_A.clamp(max=0)
            # pos_A = last_A.clamp(min=0)
            # A = d_pos * pos_A + d_neg * neg_A
            pos_A = tf.math.maximum(A, 0)
            neg_A = tf.math.minimum(A, 0)
            # d_pos = tf.expand_dims(d_pos, axis=1)
            # d_neg = tf.expand_dims(d_neg, axis=1)
            A = tf.math.multiply(pos_A, d_pos)+tf.math.multiply(neg_A, d_neg)
            bias = 0
            if b_pos is not None:
                # bias = bias + tf.einsum('sbij,bj->sbi', pos_A, b_pos)
                bias = bias + tf.tensordot(pos_A, b_pos, axes=dim)
            if b_neg is not None:
                # bias = bias + tf.einsum('sbij,bj->sbi', neg_A, b_neg)
                bias = bias + tf.tensordot(neg_A, b_neg, axes=dim)
            return A, bias
        
        if isinstance(self.shape, int):
            layer_dim = 1
        else:
            layer_dim = len(self.shape)
        dA, dbias = _bound_oneside(last_dA, dy_lower_d, dy_upper_d, dy_lower_b, dy_upper_b, layer_dim)     # lower bound only
        bias = bias + dbias
        return [dA], bias

    def eyes(self):
        if isinstance(self.shape, int):
            eye = tf.eye(self.shape)
        else:
            shape = list(self.shape)
            size = tf.reduce_prod(shape)
            eye = tf.eye(size)
            eye = tf.reshape(eye, shape+shape)
        return tf.stack([eye, -eye])