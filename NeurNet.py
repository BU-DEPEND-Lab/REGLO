import numpy as np
import pickle
import time
import io
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# import sys
# # setting path
# sys.path.append('../')
from GlobalRobust_beta import nn_extractor

from Layer import Layer, Conv2dLayer, NormalizeLayer
from CosineDecayRestarts import CosineDecayRestarts

class NeurNet:
    def __init__(self, model_name, batch_size = 1, alpha = False, beta = False, dbg_id = None):
        self.layers = []
        self.n_layers = 0
        self.model_name = model_name
        self.readyQueue = None
        self.input_shape = None
        self.input_dist_lb = None
        self.input_dist_ub = None
        self.input_lb = None
        self.input_ub = None
        self.enable_alpha = alpha
        self.enable_beta = beta
        self.batch_size = batch_size
        self.alpha_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            5e-1, decay_steps=10, decay_rate=0.8)
        # self.learning_rate = 1e-1
        self.alpha_epochs = 50
        self.epochs = 500
        # self.learning_rate = 0.005
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            0.2, decay_steps=10, decay_rate=0.8, staircase=True)
        # self.learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        #     0.5, decay_steps=10, decay_rate=2.0)
        # self.learning_rate = CosineDecayRestarts(0.4, 20, t_mul=1.0, m_mul=0.7, alpha=1e-4)
        # if dbg_id is not None:
        #     # init_rate = [0.08, 0.04, 0.02, 0.01, 0.005]
        #     # reset_steps = [10, 15, 20, 25, 30]
        #     # decay_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
        #     init_rate = [0.1, 0.08, 0.06, 0.05]
        #     reset_steps = [26, 28, 30, 32, 34]
        #     decay_rates = [0.4, 0.45, 0.5, 0.55, 0.6]
        #     i = (dbg_id//(len(decay_rates)*len(reset_steps)))
        #     j = (dbg_id//len(decay_rates)) % (len(reset_steps))
        #     k = dbg_id % (len(decay_rates))
        #     self.learning_rate = CosineDecayRestarts(init_rate[i], reset_steps[j], t_mul=1.0, m_mul=decay_rates[k], alpha=1e-4)     
        # print(self.learning_rate.get_config())
        
        self.signed_gradient_alpha = True
        self.signed_gradient_beta = False

    def setupNeuralNetwork(self, NN_layers, input_lb, input_ub, dist_lb, dist_ub):
        first_layer = NN_layers[0]
        self.input_shape = first_layer['input_shape']
        self.input_lb = tf.constant(input_lb.flatten(), dtype=tf.float32)
        self.input_ub = tf.constant(input_ub.flatten(), dtype=tf.float32)
        self.input_dist_lb = tf.constant(dist_lb.flatten(), dtype=tf.float32)
        self.input_dist_ub = tf.constant(dist_ub.flatten(), dtype=tf.float32)

        for layer_info in NN_layers:
            if layer_info['type'] == 'normalization':
                if layer_info['dim'] == 3:
                    size = np.prod(layer_info['output_shape'])
                else:
                    size = layer_info['output_shape']
                layer = NormalizeLayer(self.n_layers, size, layer_info['mean'], layer_info['std'], self.model_name, self.batch_size)
            elif layer_info['type'] == 'dense':
                layer = Layer(self.n_layers, layer_info['output_shape'], layer_info['kernel'], layer_info['bias'], self.model_name, activation=layer_info['activation'], alpha=self.enable_alpha, beta=self.enable_beta, batch_size=self.batch_size)
            elif layer_info['type'] == 'conv2d':
                layer = Conv2dLayer(self.n_layers, layer_info['input_shape'], layer_info['output_shape'], layer_info['kernel'], layer_info['bias'], layer_info['strides'], self.model_name, activation=layer_info['activation'], alpha=self.enable_alpha, beta=self.enable_beta, batch_size=self.batch_size)
            else:
                raise Exception(f"Error: layer type not supported: {layer_info['type']}")
            self.layers.append(layer)
            self.n_layers += 1
        self._init_variables()
        self.update_net_bound()

    def update_net_bound(self):
        for layer_id in range(self.n_layers):
            layer = self.layers[layer_id]
            if not layer.need_y_bounds():
                continue
            layer.y_lb = self._net_bound_backward_layer(layer_id, lb_or_ub = 1)
            layer.y_ub = self._net_bound_backward_layer(layer_id, lb_or_ub = -1)

    def _net_bound_backward_layer(self, layer_id, lb_or_ub = 1, has_bound = False):
        # lb_or_ub: 1 for lower bound, -1 for upper bound
        out_layer = self.layers[layer_id]
        A = lb_or_ub * out_layer.single_eyes()

        if has_bound:
            grad_id = layer_id + 1
        else:
            grad_id = layer_id
        self._reset_prev_layers_vars(grad_id, alpha=self.enable_alpha)
            
        def _need_pgd():
            has_alpha = False
            for j in reversed(range(grad_id)):
                layer = self.layers[j]
                if layer.need_y_bounds():
                    has_alpha = True
                    break
            return has_alpha
        if self.enable_alpha and _need_pgd():
            opt = tf.keras.optimizers.Adam(learning_rate=self.alpha_learning_rate)
            last_loss = 0.0
            stable_epochs = 0
            for epoch in range(self.alpha_epochs):
                with tf.GradientTape() as tape:
                    lb = self._net_bound_backward_one_pass(layer_id, has_bound, A)
                    loss = -tf.reduce_sum(lb)
                variables = []
                for j in range(grad_id):
                    layer = self.layers[j]
                    if not layer.need_y_bounds():
                        continue
                    variables.append(layer.alpha)
                    # if self.enable_beta:
                    #     variables.append(layer.beta)
                grads = tape.gradient(loss, variables)
                grads = [tf.math.sign(grad) for grad in grads]
                opt.apply_gradients(zip(grads, variables))
                for j in range(grad_id):
                    layer = self.layers[j]
                    if not layer.need_y_bounds():
                        continue
                    layer.clip_alpha()
                    # if self.enable_beta:
                    #     layer.clip_beta()
                loss_v = loss.numpy()
                if (abs(loss_v - last_loss) / (abs(loss_v) + 1e-9)) < 1e-3:
                    stable_epochs += 1
                else:
                    stable_epochs = 0
                if stable_epochs > 10:
                    break
                last_loss = loss_v
        lb = self._net_bound_backward_one_pass(layer_id, has_bound, A)
        return lb_or_ub * lb
    
    def _net_bound_backward_one_pass(self, layer_id, has_bound, A):
        out_layer = self.layers[layer_id]
        if has_bound:
            # if self.enable_beta:
            #     A, bias = out_layer.net_bound_backward_beta(A)
            # else:
            A, bias = out_layer.net_bound_backward(A)
        else:
            A, bias = out_layer.net_bound_backward_linear(A)
        for j in reversed(range(layer_id)):
            layer = self.layers[j]
            # if self.enable_beta:
            #     A, _b = layer.net_bound_backward_beta(A)
            # else:
            A, _b = layer.net_bound_backward(A)
            bias += _b
        lb = (
            tf.tensordot(tf.math.maximum(A, 0), self.input_lb, axes=1) + 
            tf.tensordot(tf.math.minimum(A, 0),  self.input_ub, axes=1) + 
            bias)
        return lb
    

    def _init_variables(self):
        self._compute_variable_batch_size()
        for layer in self.layers:
            # if self.enable_alpha:
            #     layer.init_alpha()
            if self.enable_beta:
                layer.init_beta()
    
    def _compute_variable_batch_size(self):
        self.n_calls = 0
        for layer in self.layers:
            if not layer.need_y_bounds():
                continue
            self.n_calls += 1
        self.n_calls += 1    # output layer call
        call_id = 0
        for layer in self.layers:
            if not layer.need_y_bounds():
                continue
            layer.set_variable_batch_size(call_id, self.n_calls)
            call_id += 1
    
    def narrow_the_dist_bound(self, fixed_alpha = False, prev_dlb = None, prev_dub = None):
        self._reset_prev_layers_vars(self.n_layers, beta=self.enable_beta)
        if self.enable_alpha:
            variables = []
            # alpha_need_clip = []
            beta_need_clip = []
            for j in range(self.n_layers):
                layer = self.layers[j]
                if not layer.need_y_bounds():
                    continue
                # if not fixed_alpha:
                #     variables.append(layer.alpha)
                #     alpha_need_clip.append(layer)
                if self.enable_beta and layer.beta_need_update():
                    variables.append(layer.beta)
                    beta_need_clip.append(layer)
            res_dlb = None
            res_dub = None
            # print("prev_bound",prev_bound)
            # opt_begin = tf.keras.optimizers.Adam(learning_rate=0.4)
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            # opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            last_loss = 0.0
            stable_epochs = 0
            loss_mean = tf.keras.metrics.Mean()
            for epoch in range(self.epochs):
                with tf.GradientTape() as tape:
                    tape.watch(variables)
                    dlb, dub = self.bound_backward()
                    loss = tf.reduce_sum(dub - dlb)
                loss_mean.update_state(loss)
                # if (epoch % (self.epochs // 5) == 0 or self.epochs - epoch <= 5):
                #     print(f"[epoch {epoch}] loss={loss_mean.result().numpy():.5f}")
                #     loss_mean.reset_states()
                
                grads = tape.gradient(loss, variables)
                loss_v = loss.numpy()
                opt.apply_gradients((grad, var) for (grad, var) in zip(grads, variables) if grad is not None)
                # for layer in alpha_need_clip:
                #     layer.clip_alpha()
                for layer in beta_need_clip:
                    layer.clip_beta()
                if res_dlb is None:
                    res_dlb = dlb
                else:
                    res_dlb = tf.math.maximum(res_dlb, dlb)
                if res_dub is None:
                    res_dub = dub
                else:
                    res_dub = tf.math.minimum(res_dub, dub)
                
                if (abs(loss_v - last_loss) / (abs(loss_v) + 1e-9)) < 1e-10:
                    break
                elif (abs(loss_v - last_loss) / (abs(loss_v) + 1e-9)) < 1e-3:
                    stable_epochs += 1
                else:
                    stable_epochs = 0
                if stable_epochs > 20:# or lowest_epochs > 50:
                    break
                last_loss = loss_v
            # if epoch > 50:
            #     print(f"[epoch {epoch+1}] loss={loss_v:.5f}, , min_loss={tf.reduce_sum(res_dub - res_dlb).numpy()}")
        return dlb, dub

    def bound_backward(self):
        for layer_id in range(self.n_layers):
            layer = self.layers[layer_id]
            if not layer.need_y_bounds():
                continue
            call_id = layer.get_layer_call_id()
            layer.dy_lb, layer.dy_ub = self._dist_bound_backward_layer(layer_id, call_id)
        out_layer = self.layers[-1]
        call_id = self.n_calls - 1  # last call
        dout_lb, dout_ub = self._dist_bound_backward_layer(self.n_layers-1, call_id, has_bound=True)
        return dout_lb, dout_ub

    def _dist_bound_backward_layer(self, layer_id, bound_layer_call_id, is_lb = True, has_bound = False):
        # lb_or_ub: 1 for lower bound, -1 for upper bound
        out_layer = self.layers[layer_id]
        # dA = (1.0 if is_lb else -1.0) * out_layer.eyes()
        dA = out_layer.eyes()
        # d_lb = self._dist_bound_backward_one_pass(layer_id, bound_layer_call_id, int(is_lb), has_bound, dA)
        # return (1.0 if is_lb else -1.0) * d_lb
        bounds = self._dist_bound_backward_one_pass(layer_id, bound_layer_call_id, int(is_lb), has_bound, dA)
        lbs, ubs = tf.split(bounds, 2)
        lbs = tf.squeeze(lbs, axis=[0])
        ubs = -tf.squeeze(ubs, axis=[0])
        return lbs, ubs
        
    def _dist_bound_backward_one_pass(self, layer_id, bound_layer_call_id, bound_id, has_bound, dA):
        out_layer = self.layers[layer_id]
        if has_bound:
            dA, bias = out_layer.dist_bound_backward(bound_layer_call_id, bound_id, dA)
        else:
            dA, bias = out_layer.dist_bound_backward_linear(dA)
        for j in reversed(range(layer_id)):
            layer = self.layers[j]
            dA, _b = layer.dist_bound_backward(bound_layer_call_id, bound_id, dA)
            bias += _b
        d_lb = (
            tf.tensordot(tf.math.maximum(dA, 0), self.input_dist_lb, axes=1) + 
            tf.tensordot(tf.math.minimum(dA, 0),  self.input_dist_ub, axes=1) + 
            bias)
        return d_lb

    def split_candidate(self):
        abs_weight = tf.ones((1, self.layers[-1].size))
        max_score = [None]*self.batch_size
        candidate = [None]*self.batch_size
        for layer_id in reversed(range(self.n_layers)):
            layer = self.layers[layer_id]
            if layer.need_y_bounds() and layer_id < self.n_layers-1:
                # layer_score = tf.reduce_sum(tf.math.multiply(abs_weight, layer.unsplit_bound()), axis=0)
                layer_score = tf.einsum('ij,bj->bj', abs_weight, layer.unsplit_bound())
                node_id = tf.math.argmax(layer_score, axis=1).numpy()
                score = tf.reduce_max(layer_score, axis = 1).numpy()
                for i in range(self.batch_size):
                    if score[i] > 0 and (max_score[i] is None or score[i] > max_score[i]):
                        max_score[i] = score[i]
                        candidate[i] = (layer_id, node_id[i])
            abs_weight = tf.matmul(abs_weight, layer.abs_weight(), transpose_b=True)
        return candidate, max_score

    def last_layer_size(self):
        return self.layers[-1].size

    def set_beta_config(self, layer_id, node_id, dist_gt_0 = True, batch_id = 0):
        self.layers[layer_id].set_beta_config(node_id, dist_gt_0, batch_id=batch_id)
    
    def reset_beta_config(self):
        for layer in self.layers:
            if layer.need_y_bounds():
                layer.reset_beta_config()
    
    def _reset_prev_layers_vars(self, layer_id, alpha=False, beta=False):
        for layer in self.layers[:layer_id]:
            if layer.need_y_bounds():
                if alpha:
                    layer.reset_alpha()
                if beta:
                    layer.reset_beta()



def load_model():
    netname = 'model_mnist_2c2d'
    json_filename = 'data/model/' + netname + '.json'
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    dnn_model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    # load weights into new model
    dnn_model.load_weights('data/model/' + netname + '.h5')
    # extract model infos
    return nn_extractor(dnn_model), netname

    

def load_extracted_model(fname):
    netname = 'model_cifar10_' + fname
    pickle_filename = 'data/model/' + netname + '.pickle'
    extracted_layers = None
    with open(pickle_filename, 'rb') as f:
        extracted_layers = pickle.load(f)
    return extracted_layers

def main():
    # model_name = '4c2d_noBN_4c2d_reg_1e_3_0'
    # input_lb = imageio.imread("data/cifar_lb.png") / 255.0
    # input_ub = imageio.imread("data/cifar_ub.png") / 255.0
    # print(np.max(input_ub), np.min(input_lb))

    # eps = 1.0/255.0
    eps = 1e-3
    # input_lb = np.clip(input_lb - eps, 0.0, 1.0)
    # input_ub = np.clip(input_ub + eps, 0.0, 1.0)
    input_lb = np.zeros((28,28,1))
    input_ub = np.ones((28,28,1))
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    # Layers = load_extracted_model(model_name)
    Layers, model_name = load_model()
    test_NN = NeurNet(model_name, alpha=True)
    test_NN.setupNeuralNetwork(Layers, input_lb, input_ub, diff_lb, diff_ub)
    lb, ub, dlb, dub = test_NN.bound_backward()
    print('output bound:', lb.numpy(), ub.numpy())
    print('output variation bound:', dlb.numpy(), dub.numpy())

if __name__ == '__main__':
    main()


