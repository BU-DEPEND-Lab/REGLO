from enum import Enum, auto
import tensorflow as tf
import numpy as np

class Activation(Enum):
    LINEAR = auto()
    RELU = auto()

class Layer:
    def __init__(self, layer_id, size, weight, bias, model_name, activation='linear', alpha = False, beta = False):
        self.size = size
        self.layer_id = layer_id
        self.model_name = model_name
        self.activation = Activation.LINEAR
        if activation == 'relu':
            self.activation = Activation.RELU
        self.weight = tf.constant(weight, dtype = tf.float32)
        self.bias = tf.constant(bias, dtype = tf.float32)
        self.y_lb = None
        self.y_ub = None
        self.dy_lb = None
        self.dy_ub = None
        self.alpha = None
        self.enable_alpha = alpha
        if self.enable_alpha:
            self.alpha = tf.Variable(0.5*tf.ones(self.size), dtype=tf.float32)
        # self.beta = None
        # self.gamma = None
        # self.enable_beta = beta
        # if self.enable_beta:
        #     self.beta = tf.Variable(tf.zeros(self.size), dtype=tf.float32)
        #     self.gamma = tf.Variable(tf.zeros(self.size), dtype=tf.float32)
        #     self.init_beta_config()

    def dist_bound_backward(self, last_dA):
        # if self.enable_beta:
        #     return self.dist_bound_backward_beta(self, last_A, last_dA)
        dA = last_dA
        bias = 0
        if self.activation == Activation.RELU:
            dy_lb = tf.math.minimum(self.dy_lb, 0)
            dy_ub = tf.math.maximum(self.dy_ub, 0)
            dy_ub = tf.math.maximum(dy_ub, dy_lb + 1e-8)
            dy_lb_l = tf.math.minimum(tf.math.maximum(-self.y_ub, self.dy_lb), 0)
            dy_ub_l = tf.math.maximum(tf.math.minimum(-self.y_lb, self.dy_ub), 0)
            dy_ub_l = tf.math.maximum(dy_ub_l, dy_lb_l + 1e-8)

            dy_diff_u = dy_ub - dy_lb
            dy_diff_l = dy_ub_l - dy_lb_l

            dx_lb_u = tf.math.minimum(tf.math.maximum(dy_lb, -self.y_lb), 0)        # new upper bound
            dx_ub_u = tf.math.maximum(dy_ub + tf.minimum(self.y_ub, 0), 0)
            dx_diff_u = dx_ub_u - dx_lb_u
            dy_upper_d = tf.math.divide(dx_diff_u, dy_diff_u)
            dy_upper_b = dx_lb_u + tf.math.multiply(-dy_lb, dy_upper_d)

            # dy_upper_d = tf.math.divide(dy_ub, dy_diff_u)     # Old upper bound
            # dy_upper_b = tf.math.multiply(-dy_lb, dy_upper_d)

            dy_lower_d = tf.math.divide(-dy_lb_l, dy_diff_l)
            dy_lower_b = tf.math.multiply(-dy_ub_l, dy_lower_d)

            def _bound_oneside(A, d_pos, d_neg, b_pos, b_neg):
                # multiply according to sign of A (we use fused operation to save memory)
                # neg_A = last_A.clamp(max=0)
                # pos_A = last_A.clamp(min=0)
                # A = d_pos * pos_A + d_neg * neg_A
                pos_A = tf.math.maximum(A, 0)
                neg_A = tf.math.minimum(A, 0)
                A = tf.math.multiply(pos_A, d_pos)+tf.math.multiply(neg_A, d_neg)
                bias = 0
                if b_pos is not None:
                    # bias = bias + tf.einsum('sb...,sb...->sb', pos_A, b_pos)
                    bias = bias + tf.tensordot(pos_A, b_pos, axes=1)
                if b_neg is not None:
                    # bias = bias + tf.einsum('sb...,sb...->sb', neg_A, b_neg)
                    bias = bias + tf.tensordot(neg_A, b_neg, axes=1)
                return A, bias
            
            dA, dbias = _bound_oneside(last_dA, dy_lower_d, dy_upper_d, dy_lower_b, dy_upper_b)     # lower bound only
            bias = bias + dbias
        dA, linear_bias = self.dist_bound_backward_linear(dA) # linear_dbias = 0
        bias = bias + linear_bias
        # dbias = dbias + linear_dbias # linear_dbias = 0
        return dA, bias

    def dist_bound_backward_linear(self, last_dA):
        # A = tf.matmul(last_A, self.weight, transpose_b=True)
        # bias = tf.tensordot(last_A, self.bias, axes=1)
        dA = tf.matmul(last_dA, self.weight, transpose_b=True)
        dbias = 0
        return dA, dbias

    def net_bound_backward(self, last_A):
        # if self.enable_beta:
        #     return self.net_bound_backward_beta(last_A)
        A, bias = last_A, 0
        if self.activation == Activation.RELU:
            y_lb = tf.math.minimum(self.y_lb, 0)
            y_ub = tf.math.maximum(self.y_ub, 0)
            y_ub = tf.math.maximum(y_ub, y_lb + 1e-8)

            y_diff = y_ub - y_lb
            y_upper_d = tf.math.divide(y_ub, y_diff)
            y_upper_b = tf.math.multiply(-y_lb, y_upper_d)
            if self.enable_alpha:
                y_lower_d = self.alpha
            else:
                y_lower_d = tf.cast(y_upper_d > 0.5, dtype = tf.float32)
            # y_lower_b = tf.zeros_like(y_upper_b)      # we can set y_lower_b = None in _bound_oneside()

            def _bound_oneside(A, d_pos, d_neg, b_pos, b_neg):
                # multiply according to sign of A (we use fused operation to save memory)
                # neg_A = last_A.clamp(max=0)
                # pos_A = last_A.clamp(min=0)
                # A = d_pos * pos_A + d_neg * neg_A
                pos_A = tf.math.maximum(A, 0)
                neg_A = tf.math.minimum(A, 0)
                A = tf.math.multiply(pos_A, d_pos)+tf.math.multiply(neg_A, d_neg)
                bias = 0
                if b_pos is not None:
                    # bias = bias + tf.einsum('sb...,sb...->sb', pos_A, b_pos)
                    bias = bias + tf.tensordot(pos_A, b_pos, axes=1)
                if b_neg is not None:
                    # bias = bias + tf.einsum('sb...,sb...->sb', neg_A, b_neg)
                    bias = bias + tf.tensordot(neg_A, b_neg, axes=1)
                return A, bias
            
            A, bias = _bound_oneside(last_A, y_lower_d, y_upper_d, None, y_upper_b)     # lower bound only
        A, linear_bias = self.net_bound_backward_linear(A) # linear_dbias = 0
        bias = bias + linear_bias
        return A, bias

    def net_bound_backward_linear(self, last_A):
        A = tf.matmul(last_A, self.weight, transpose_b=True)
        bias = tf.tensordot(last_A, self.bias, axes=1)
        return A, bias
    
    def eyes(self):
        return tf.eye(self.size)

    def need_y_bounds(self):
        return self.activation == Activation.RELU

    def update_alpha(self, grad, learning_rate, signed = False):
        if signed:
            grad = tf.math.sign(grad)
        self.alpha.assign_add(learning_rate * grad)
        self.clip_alpha()

    def clip_alpha(self):
        self.alpha.assign(tf.clip_by_value(self.alpha, 0.0, 1.0))

    # def init_beta_config(self):
    #     self.beta_coef = tf.Variable(tf.zeros(self.size), trainable=False)
    #     self.gamma_coef = tf.Variable(tf.zeros(self.size), trainable=False)
    #     self.split_condition = []
    #     for i in range(4):
    #         self.split_condition.append(tf.Variable(tf.zeros(self.size), trainable=False))
    #     self.split_condition.append(tf.Variable(tf.ones(self.size), trainable=False))
    
    # def reset_beta_config(self):
    #     self.beta_coef.assign(tf.zeros(self.size))
    #     self.gamma_coef.assign(tf.zeros(self.size))
    #     for i in range(4):
    #         self.split_condition[i].assign(tf.zeros(self.size))
    #     self.split_condition[4].assign(tf.ones(self.size))

    def reset_alpha(self):
        self.alpha.assign(0.5*tf.ones(self.size))
    # def reset_beta(self):
    #     self.beta.assign(tf.zeros(self.size))
    # def reset_gamma(self):
    #     self.gamma.assign(tf.zeros(self.size))

    # def set_beta_config(self, node, neuron_gt_0 = True, dist_gt_0 = True):
    #     self.beta_coef[node].assign(-1.0 if neuron_gt_0 else 1.0)
    #     self.gamma_coef[node].assign(-1.0 if dist_gt_0 else 1.0)
    #     if neuron_gt_0:
    #         if dist_gt_0:   # condition 0 - y+dy > 0 && y > 0
    #             self.split_condition[0][node].assign(1.0)
    #         else:           # condition 3 - y+dy < 0 && y > 0
    #             self.split_condition[3][node].assign(1.0)
    #     else:
    #         if dist_gt_0:   # condition 2 - y+dy > 0 && y < 0
    #             self.split_condition[2][node].assign(1.0)
    #         else:           # condition 1 - y+dy < 0 && y < 0
    #             self.split_condition[1][node].assign(1.0)
    #     self.split_condition[4][node].assign(0.0)       # condition 4 - not splited

    # def update_beta(self, grad, learning_rate, signed = False):
    #     if signed:
    #         grad = tf.math.sign(grad)
    #     self.beta.assign_add(learning_rate * grad)
    #     self.beta.assign(tf.clip_by_value(self.beta, 0.0, tf.float32.max))

    # def update_gamma(self, grad, learning_rate, signed = False):
    #     if signed:
    #         grad = tf.math.sign(grad)
    #     self.gamma.assign_add(learning_rate * grad)
    #     self.gamma.assign(tf.clip_by_value(self.gamma, 0.0, tf.float32.max))
    
    # def clip_beta(self):
    #     self.beta.assign(tf.clip_by_value(self.beta, 0.0, tf.float32.max))
    
    # def clip_gamma(self):
    #     self.gamma.assign(tf.clip_by_value(self.gamma, 0.0, tf.float32.max))

    # def abs_weight(self):
    #     abs_weight = tf.abs(self.weight)
    #     if len(abs_weight.shape) == 1:
    #         abs_weight = tf.linalg.diag(abs_weight)
    #     return abs_weight

    # def unsplit_bound(self):
    #     return tf.math.multiply(self.split_condition[4], tf.math.maximum(tf.abs(self.dy_lb), tf.abs(self.dy_ub)))

    # def __iter__(self):
    #     self._iter_i = 0
    #     return self
    
    # def __next__(self):
    #     if self._iter_i >= self.size:
    #         raise StopIteration
    #     node = self.getNode(self._iter_i)
    #     self._iter_i += 1
    #     return node
    
    
class Conv2dLayer(Layer):
    def __init__(self, layer_id, input_shape, output_shape, weight, bias, strides, model_name, activation='linear', alpha = False, beta = False):
        self.w_sz = output_shape[0]
        self.h_sz = output_shape[1]
        self.c_sz = output_shape[2]
        self.strides = strides
        weight, bias = self._unrollWeights(input_shape, output_shape, weight, bias, strides)
        super().__init__(layer_id, self.w_sz*self.h_sz*self.c_sz, weight, bias, model_name, activation, alpha, beta)

    @staticmethod
    def _unrollWeights(input_shape, output_shape, w, b, strides):
        unrolled_w = np.zeros((np.prod(input_shape), np.prod(output_shape)))
        unrolled_b = np.zeros(np.prod(output_shape))
        knsz_w, knsz_h, cin, cout = w.shape
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                for k in range(output_shape[2]):
                    out_idx = (i * output_shape[1] + j) * output_shape[2] + k
                    unrolled_b[out_idx] = b[k]
                    for l in range(knsz_w):
                        for m in range(knsz_h):
                            for n in range(cin):
                                in_i, in_j, in_k = strides[0] * i + l, strides[1] * j + m, n
                                in_idx = (in_i * input_shape[1] + in_j) * input_shape[2] + in_k
                                unrolled_w[in_idx, out_idx] = w[l, m, n, k]
        return unrolled_w, unrolled_b


class NormalizeLayer(Layer):
    def __init__(self, layer_id, size, mean, std, model_name):
        std = np.maximum(std, 1e-10)
        norm_weight = 1/std
        norm_bias = -mean/std
        super().__init__(layer_id, size, norm_weight, norm_bias, model_name)

    def dist_bound_backward_linear(self, last_dA):
        # A = tf.math.multiply(last_A, self.weight)
        # bias = tf.tensordot(last_A, self.bias, axes=1)
        dA = tf.math.multiply(last_dA, self.weight)
        dbias = 0
        return dA, dbias

    def net_bound_backward_linear(self, last_A):
        A = tf.math.multiply(last_A, self.weight)
        bias = tf.tensordot(last_A, self.bias, axes=1)
        return A, bias