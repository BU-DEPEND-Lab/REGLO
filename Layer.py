from enum import Enum, auto
import tensorflow as tf
import numpy as np

class Activation(Enum):
    LINEAR = auto()
    RELU = auto()

class Layer:
    def __init__(self, layer_id, size, weight, bias, model_name, activation='linear', alpha = False, beta = False, batch_size = 1):
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
        self.beta = None
        self.enable_beta = beta
        self.beta_activated = False
        self.var_batch_size = 0
        self.call_id = layer_id     # the i-th layer to be called to evaluate y bounds
        self.batch_size = batch_size

    def _dist_linear_bounds(self, consider_splits=False) -> list:
        dy_lb = tf.math.minimum(self.dy_lb, 0)
        dy_ub = tf.math.maximum(self.dy_ub, 0)
        dy_ub = tf.math.maximum(dy_ub, dy_lb + 1e-8)

        dy_lb_l = tf.math.minimum(tf.math.maximum(-self.y_ub, self.dy_lb), 0)        # new lower and upper bounds
        dy_ub_l = tf.math.maximum(tf.math.minimum(-self.y_lb, self.dy_ub), 0)
        dy_ub_l = tf.math.maximum(dy_ub_l, dy_lb_l + 1e-8)

        dy_diff_u = dy_ub - dy_lb
        dy_diff_l = dy_ub_l - dy_lb_l
        dy_lower_d = tf.math.divide(-dy_lb_l, dy_diff_l)
        dy_lower_b = tf.math.multiply(-dy_ub_l, dy_lower_d)

        dx_lb_u = tf.math.minimum(tf.math.maximum(dy_lb, -self.y_lb), 0)
        dx_ub_u = tf.math.maximum(dy_ub + tf.minimum(self.y_ub, 0), 0)
        dx_diff_u = dx_ub_u - dx_lb_u
        dy_upper_d = tf.math.divide(dx_diff_u, dy_diff_u)
        dy_upper_b = dx_lb_u + tf.math.multiply(-dy_lb, dy_upper_d)

        # dy_diff = dy_ub - dy_lb                               # old lower and upper bounds
        # dy_upper_d = tf.math.divide(dy_ub, dy_diff)
        # dy_upper_b = tf.math.multiply(-dy_lb, dy_upper_d)
        # dy_lower_d = tf.math.divide(-dy_lb, dy_diff)
        # dy_lower_b = tf.math.multiply(-dy_ub, dy_lower_d)

        res = [dy_lower_d, dy_lower_b, dy_upper_d, dy_upper_b]
        
        if consider_splits:
            geq0_dy_lower_d = tf.cast(self.y_lb > 0, dtype=tf.float32)
            geq0_dy_upper_d = tf.math.divide(tf.math.maximum(dx_ub_u, 1e-16), tf.math.maximum(dy_ub, 1e-16))
            leq0_dy_lower_d = tf.cast(self.y_ub >= 0, dtype=tf.float32)
            leq0_dy_upper_d = tf.math.divide(dx_lb_u, tf.math.minimum(dy_lb, -1e-16))
            
            # geq0_dy_lower_d = 0     # old lower and upper bounds
            # geq0_dy_upper_d = 1
            # leq0_dy_lower_d = 1
            # leq0_dy_upper_d = 0

            res.append(geq0_dy_lower_d)
            res.append(geq0_dy_upper_d)
            res.append(leq0_dy_lower_d)
            res.append(leq0_dy_upper_d)

        return res


    def dist_bound_backward(self, bound_layer_call_id, bound_id, last_dA):
        if self.enable_beta and self.beta_activated:
            return self.dist_bound_backward_beta(bound_layer_call_id, bound_id, last_dA)
        dA = last_dA
        bias = 0
        if self.activation == Activation.RELU:
            # if self.enable_alpha:
            #     alpha = self.alpha[bound_id, bound_layer_call_id - self.call_id - 1]
            
            tmp = self._dist_linear_bounds()
            dy_lower_d, dy_lower_b, dy_upper_d, dy_upper_b = tmp[0], tmp[1], tmp[2], tmp[3]

            def _bound_oneside(A, d_pos, d_neg, b_pos, b_neg):
                # multiply according to sign of A (we use fused operation to save memory)
                # neg_A = last_A.clamp(max=0)
                # pos_A = last_A.clamp(min=0)
                # A = d_pos * pos_A + d_neg * neg_A
                pos_A = tf.math.maximum(A, 0)
                neg_A = tf.math.minimum(A, 0)
                d_pos = tf.expand_dims(d_pos, axis=1)
                d_neg = tf.expand_dims(d_neg, axis=1)
                A = tf.math.multiply(pos_A, d_pos)+tf.math.multiply(neg_A, d_neg)
                bias = 0
                if b_pos is not None:
                    bias = bias + tf.einsum('sbij,bj->sbi', pos_A, b_pos)
                    # bias = bias + tf.tensordot(pos_A, b_pos, axes=1)
                if b_neg is not None:
                    bias = bias + tf.einsum('sbij,bj->sbi', neg_A, b_neg)
                    # bias = bias + tf.tensordot(neg_A, b_neg, axes=1)
                return A, bias
            
            dA, dbias = _bound_oneside(last_dA, dy_lower_d, dy_upper_d, dy_lower_b, dy_upper_b)     # lower bound only
            bias = bias + dbias
        dA, linear_bias = self.dist_bound_backward_linear(dA) # linear_dbias = 0
        bias = bias + linear_bias
        return dA, bias

    def dist_bound_backward_linear(self, last_dA):
        dA = tf.matmul(last_dA, self.weight, transpose_b=True)
        bias = 0
        return dA, bias

    def dist_bound_backward_beta(self, bound_layer_call_id, bound_id, last_dA):
        dA = last_dA
        bias = 0
        if self.activation == Activation.RELU:
            # if self.enable_alpha:
            #     alpha = self.alpha[bound_id, bound_layer_call_id - self.call_id - 1]
            beta = self.beta[bound_layer_call_id - self.call_id - 1]
            tmp = self._dist_linear_bounds(consider_splits=True)
            dy_lower_d, dy_lower_b, dy_upper_d, dy_upper_b = tmp[0], tmp[1], tmp[2], tmp[3]
            geq0_dy_lower_d = tmp[4]
            geq0_dy_upper_d = tmp[5]
            leq0_dy_lower_d = tmp[6]
            leq0_dy_upper_d = tmp[7]

            dy_upper_d = (tf.math.multiply(self.split_condition[0], geq0_dy_upper_d) + 
                          tf.math.multiply(self.split_condition[1], leq0_dy_upper_d) + 
                          tf.math.multiply(self.split_condition[2], dy_upper_d))
            dy_upper_b = tf.math.multiply(self.split_condition[2], dy_upper_b)
            dy_lower_d = (tf.math.multiply(self.split_condition[0], geq0_dy_lower_d) + 
                          tf.math.multiply(self.split_condition[1], leq0_dy_lower_d) + 
                          tf.math.multiply(self.split_condition[2], dy_lower_d))
            dy_lower_b = tf.math.multiply(self.split_condition[2], dy_lower_b)

            def _bound_oneside(A, d_pos, d_neg, b_pos, b_neg):
                pos_A = tf.math.maximum(A, 0)
                neg_A = tf.math.minimum(A, 0)
                d_pos = tf.expand_dims(d_pos, axis=1)
                d_neg = tf.expand_dims(d_neg, axis=1)
                A = tf.math.multiply(pos_A, d_pos)+tf.math.multiply(neg_A, d_neg)
                bias = 0
                if b_pos is not None:
                    bias = bias + tf.einsum('sbij,bj->sbi', pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + tf.einsum('sbij,bj->sbi', neg_A, b_neg)
                return A, bias
                
            dA, dbias = _bound_oneside(last_dA, dy_lower_d, dy_upper_d, dy_lower_b, dy_upper_b)     # lower bound only
            dA = dA + tf.expand_dims(tf.math.multiply(beta, self.beta_coef), axis=2)
            bias = bias + dbias
            
        dA, linear_bias = self.dist_bound_backward_linear(dA) # linear_dbias = 0
        bias = bias + linear_bias
        return dA, bias

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
        eye = tf.eye(self.size, batch_shape=[1,self.batch_size])
        return tf.concat([eye, -eye], 0)

    def single_eyes(self):  # for net bound propagation (alpha CROWN)
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

    # def init_alpha(self):
    #     self.alpha = tf.Variable(0.5*tf.ones((2, self.var_batch_size, self.size)), dtype=tf.float32, trainable=False)

    def init_beta(self):
        self.beta = tf.Variable(tf.zeros((self.var_batch_size, 2, self.batch_size, self.size)), dtype=tf.float32, trainable=False)
        self.beta_coef = tf.Variable(tf.zeros((self.batch_size, self.size)), trainable=False)
        self.split_condition = []
        for i in range(2):
            self.split_condition.append(tf.Variable(tf.zeros((self.batch_size, self.size)), trainable=False))
        self.split_condition.append(tf.Variable(tf.ones((self.batch_size, self.size)), trainable=False))
        self.beta_activated = False
    
    def reset_beta_config(self):
        self.beta_coef.assign(tf.zeros((self.batch_size, self.size)))
        for i in range(2):
            self.split_condition[i].assign(tf.zeros((self.batch_size, self.size)))
        self.split_condition[2].assign(tf.ones((self.batch_size, self.size)))
        self.beta_activated = False

    def reset_alpha(self):
        self.alpha.assign(0.5*tf.ones(self.size))

    def reset_beta(self):
        self.beta.assign(tf.zeros((self.var_batch_size, 2, self.batch_size, self.size)))
        # self.beta.trainable = False

    def set_beta_config(self, node, dist_gt_0 = True, batch_id = 0):
        self.beta_coef[batch_id, node].assign(-1.0 if dist_gt_0 else 1.0)
        if dist_gt_0:   # condition 0: dy > 0
            self.split_condition[0][batch_id, node].assign(1.0)
        else:           # condition 1: dy < 0
            self.split_condition[1][batch_id, node].assign(1.0)
        self.split_condition[2][batch_id, node].assign(0.0)       # condition 2: not splited
        self.beta_activated = True

    def beta_need_update(self):
        return self.beta_activated

    def update_beta(self, grad, learning_rate, signed = False):
        if signed:
            grad = tf.math.sign(grad)
        self.beta.assign_add(learning_rate * grad)
        self.clip_beta()
    
    def clip_beta(self):
        self.beta.assign(tf.clip_by_value(self.beta, 0.0, tf.float32.max))

    # def set_alpha_trainable(self, trainable):
    #     self.alpha.trainable = trainable

    def set_beta_trainable(self, trainable):
        self.beta.trainable = trainable

    def abs_weight(self):
        abs_weight = tf.abs(self.weight)
        if len(abs_weight.shape) == 1:
            abs_weight = tf.linalg.diag(abs_weight)
        return abs_weight

    def unsplit_bound(self):
        effective = tf.cast(tf.math.logical_and(tf.math.less(self.dy_lb, 0), tf.math.less(0, self.dy_ub)), tf.float32)
        worstbounds = tf.math.multiply(effective, tf.math.maximum(-self.dy_lb, self.dy_ub))
        return tf.math.multiply(self.split_condition[2], worstbounds)

    def set_variable_batch_size(self, call_id, n_calls):
        self.call_id = call_id
        self.var_batch_size = n_calls - self.call_id - 1    # the variables will be needed when the subsequent layers are evaluating y bounds

    def get_layer_call_id(self):
        return self.call_id
    
    
class Conv2dLayer(Layer):
    def __init__(self, layer_id, input_shape, output_shape, weight, bias, strides, model_name, activation='linear', alpha = False, beta = False, batch_size = 1):
        self.w_sz = output_shape[0]
        self.h_sz = output_shape[1]
        self.c_sz = output_shape[2]
        self.strides = strides
        weight, bias = self._unrollWeights(input_shape, output_shape, weight, bias, strides)
        super().__init__(layer_id, self.w_sz*self.h_sz*self.c_sz, weight, bias, model_name, activation, alpha, beta, batch_size=batch_size)

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
    def __init__(self, layer_id, size, mean, std, model_name, batch_size = 1):
        std = np.maximum(std, 1e-10)
        mean = mean.flatten()
        std = std.flatten()
        norm_weight = 1/std
        norm_bias = -mean/std
        super().__init__(layer_id, size, norm_weight, norm_bias, model_name, batch_size=batch_size)

    def dist_bound_backward_linear(self, last_dA):
        dA = tf.math.multiply(last_dA, self.weight)
        bias = 0
        return dA, bias

    def net_bound_backward_linear(self, last_A):
        A = tf.math.multiply(last_A, self.weight)
        bias = tf.tensordot(last_A, self.bias, axes=1)
        return A, bias