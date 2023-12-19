import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
# import imageio
# from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# setting path
import sys
from os.path import dirname, abspath
sys.path.append(dirname(abspath(__file__)))
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from GlobalRobust_beta import nn_extractor

from itnecrown_Layer import Layer, Conv2dLayer, NormalizeLayer

class NeurNet:
    def __init__(self, model_name, alpha = False, beta = False):
        self.layers = []
        self.n_layers = 0
        self.model_name = model_name
        self.readyQueue = None
        self.input_shape = None
        self.input_lb = None
        self.input_ub = None
        self.input_dist_lb = None
        self.input_dist_ub = None
        self.enable_alpha = alpha
        self.enable_beta = False
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            5e-1, decay_steps=10, decay_rate=0.8)
        # self.learning_rate = 1e-1
        self.epochs = 50
        self.signed_gradient_alpha = True
        self.signed_gradient_beta = False

    def setupNeuralNetwork(self, NN_layers, input_lb=None, input_ub=None, dist_lb=None, dist_ub=None):
        first_layer = NN_layers[0]
        self.input_shape = first_layer['input_shape']

        if input_lb is not None:
            self.setupBounds(input_lb, input_ub, dist_lb, dist_ub)

        # if first_layer['type'] == 'conv2d' or (first_layer['type'] == 'normalization' and first_layer['dim'] == 3):
        #     input_layer = Conv2dLayer(0, first_layer['input_shape'], None, self.model_name)
        # else:
        #     input_layer = Layer(0, first_layer['input_shape'], self.model_name)
        # input_layer.setupLayer()
        # input_layer.setBounds(input_lb, input_ub, dist_lb, dist_ub)
        # self.layers.append(input_layer)
        # self.n_layers = 1
        for layer_info in NN_layers:
            if layer_info['type'] == 'normalization':
                if layer_info['dim'] == 3:
                    size = np.prod(layer_info['output_shape'])
                else:
                    size = layer_info['output_shape']
                layer = NormalizeLayer(self.n_layers, size, layer_info['mean'], layer_info['std'], self.model_name)
            elif layer_info['type'] == 'dense':
                layer = Layer(self.n_layers, layer_info['output_shape'], layer_info['kernel'], layer_info['bias'], self.model_name, activation=layer_info['activation'], alpha=self.enable_alpha, beta=self.enable_beta)
            elif layer_info['type'] == 'conv2d':
                layer = Conv2dLayer(self.n_layers, layer_info['input_shape'], layer_info['output_shape'], layer_info['kernel'], layer_info['bias'], layer_info['strides'], self.model_name, activation=layer_info['activation'], alpha=self.enable_alpha, beta=self.enable_beta)
            else:
                raise Exception(f"Error: layer type not supported: {layer_info['type']}")
            self.layers.append(layer)
            self.n_layers += 1

    def setupNNfromTFmodel(self, model, ignore_output_layer=False, input_lb=None, input_ub=None, dist_lb=None, dist_ub=None):
        NN_Layers = nn_extractor(model)
        if ignore_output_layer:
            NN_Layers = NN_Layers[:-1]
        self.setupNeuralNetwork(NN_Layers, input_lb, input_ub, dist_lb, dist_ub)
        
    def setupBounds(self, input_lb, input_ub, dist_lb, dist_ub):
        self.input_lb = tf.constant(input_lb.flatten(), dtype=tf.float32)
        self.input_ub = tf.constant(input_ub.flatten(), dtype=tf.float32)
        self.input_dist_lb = tf.constant(dist_lb.flatten(), dtype=tf.float32)
        self.input_dist_ub = tf.constant(dist_ub.flatten(), dtype=tf.float32)

    def bound_backward(self):
        for layer_id in range(self.n_layers):
            layer = self.layers[layer_id]
            if not layer.need_y_bounds():
                continue
            layer.y_lb = self._net_bound_backward_layer(layer_id, lb_or_ub = 1)
            layer.y_ub = self._net_bound_backward_layer(layer_id, lb_or_ub = -1)
            layer.dy_lb, _, _ = self._dist_bound_backward_layer(layer_id, lb_or_ub = 1)
            layer.dy_ub, _, _ = self._dist_bound_backward_layer(layer_id, lb_or_ub = -1)
            # print(layer_id, tf.reduce_min(layer.y_ub).numpy(), tf.reduce_max(layer.y_ub).numpy())
        out_layer = self.layers[-1]
        out_lb = self._net_bound_backward_layer(self.n_layers-1, lb_or_ub = 1, has_bound=True)
        out_ub = self._net_bound_backward_layer(self.n_layers-1, lb_or_ub = -1, has_bound=True)
        dout_lb, dA_lb, bias_lb = self._dist_bound_backward_layer(self.n_layers-1, lb_or_ub = 1, has_bound=True)
        dout_ub, dA_ub, bias_ub = self._dist_bound_backward_layer(self.n_layers-1, lb_or_ub = -1, has_bound=True)
        return out_lb, out_ub, dout_lb, dout_ub, dA_lb, dA_ub, bias_lb, bias_ub

    def _net_bound_backward_layer(self, layer_id, lb_or_ub = 1, has_bound = False):
        # lb_or_ub: 1 for lower bound, -1 for upper bound
        out_layer = self.layers[layer_id]
        A = lb_or_ub * out_layer.eyes()

        if has_bound:
            grad_id = layer_id + 1
        else:
            grad_id = layer_id
        self._reset_prev_layers_vars(grad_id, alpha=self.enable_alpha, beta=self.enable_beta)
            
        def _need_pgd():
            has_alpha = False
            for j in reversed(range(grad_id)):
                layer = self.layers[j]
                if layer.need_y_bounds():
                    has_alpha = True
                    break
            return has_alpha
        if self.enable_alpha and _need_pgd():
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            last_loss = 0.0
            stable_epochs = 0
            for epoch in range(self.epochs):
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
    
    def _dist_bound_backward_layer(self, layer_id, lb_or_ub = 1, has_bound = False):
        # lb_or_ub: 1 for lower bound, -1 for upper bound
        out_layer = self.layers[layer_id]
        dA = lb_or_ub * out_layer.eyes()
        # A = tf.zeros_like(dA)

        # if has_bound:
        #     grad_id = layer_id + 1
        # else:
        #     grad_id = layer_id
        # self._reset_prev_layers_vars(grad_id, alpha=self.enable_alpha, beta=self.enable_beta, gamma=self.enable_beta)
            
        # def _need_pgd():
        #     has_alpha = False
        #     for j in reversed(range(grad_id)):
        #         layer = self.layers[j]
        #         if layer.need_y_bounds():
        #             has_alpha = True
        #             break
        #     return has_alpha

        # if self.enable_alpha and _need_pgd():
        #     opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #     last_loss = 0.0
        #     stable_epochs = 0
        #     for epoch in range(self.epochs):
        #         with tf.GradientTape(persistent=True) as tape:
        #             d_lb = self._dist_bound_backward_one_pass(layer_id, has_bound, dA)
        #             loss = -tf.reduce_sum(d_lb)     
        #             # use Adam for gradient descent (since we need gradient ascent of d_lb, 
        #             # we gradient descent the negetive loss)
        #         variables = []
        #         for j in range(grad_id):
        #             layer = self.layers[j]
        #             if not layer.need_y_bounds():
        #                 continue
        #             variables.append(layer.alpha)
        #             # if self.enable_beta:
        #             #     variables.append(layer.beta)
        #             #     variables.append(layer.gamma)
        #                 # if j == 2 and (epoch % (self.epochs // 10) == 0 or self.epochs - epoch <= 5):
        #                 #     print(f"[epoch {epoch+1}] beta={layer.beta[0].numpy():.5f}, gamma={layer.gamma[0].numpy():.5f}, loss={-lb_or_ub * loss.numpy():.5f}")
        #         grads = tape.gradient(loss, variables)
        #         opt.apply_gradients(zip(grads, variables))
        #         for j in range(grad_id):
        #             layer = self.layers[j]
        #             if not layer.need_y_bounds():
        #                 continue
        #             layer.clip_alpha()
        #             # if self.enable_beta:
        #             #     layer.clip_beta()
        #             #     layer.clip_gamma()
        #         loss_v = loss.numpy()
        #         if (abs(loss_v - last_loss) / (abs(loss_v) + 1e-9)) < 1e-3:
        #             stable_epochs += 1
        #         else:
        #             stable_epochs = 0
        #         if stable_epochs > 10:
        #             break
        #         last_loss = loss_v
                
        d_lb, dA, bias = self._dist_bound_backward_one_pass(layer_id, has_bound, dA)
        # if layer_id==3:
        #     print(d_lb.numpy())
        
        return lb_or_ub * d_lb, lb_or_ub * dA, lb_or_ub * bias
        
    def _dist_bound_backward_one_pass(self, layer_id, has_bound, dA):
        out_layer = self.layers[layer_id]
        if has_bound:
            # if self.enable_beta:
            #     dA, bias = out_layer.dist_bound_backward_beta(dA)
            # else:
            dA, bias = out_layer.dist_bound_backward(dA)
        else:
            dA, bias = out_layer.dist_bound_backward_linear(dA)
        for j in reversed(range(layer_id)):
            layer = self.layers[j]
            # if self.enable_beta:
            #     dA, _b = layer.dist_bound_backward_beta(dA)
            # else:
            dA, _b = layer.dist_bound_backward(dA)
            bias += _b
        d_lb = (
            tf.tensordot(tf.math.maximum(dA, 0), self.input_dist_lb, axes=1) + 
            tf.tensordot(tf.math.minimum(dA, 0),  self.input_dist_ub, axes=1) + 
            bias)
        return d_lb, dA, bias


    def last_layer_size(self):
        return self.layers[-1].size

    # def set_beta_config(self, layer_id, node_id, neuron_gt_0 = True, dist_gt_0 = True):
    #     self.layers[layer_id].set_beta_config(node_id, neuron_gt_0, dist_gt_0)
    
    # def reset_beta_config(self):
    #     for layer in self.layers:
    #         if layer.need_y_bounds():
    #             layer.reset_beta_config()
    
    def _reset_prev_layers_vars(self, layer_id, alpha=True, beta=False, gamma=False):
        for layer in self.layers[:layer_id]:
            if layer.need_y_bounds():
                if alpha:
                    layer.reset_alpha()
                # if beta:
                #     layer.reset_beta()
                # if gamma:
                #     layer.reset_gamma()



def load_model():
    netname = 'model_mnist_2c2d'
    directory = dirname(dirname(dirname(abspath(__file__)))) + '/data/model/'
    json_filename = directory + netname + '.json'
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    dnn_model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    # load weights into new model
    dnn_model.load_weights(directory + netname + '.h5')
    # extract model infos
    return nn_extractor(dnn_model), netname

    

def load_extracted_model(fname):
    netname = 'model_cifar10_' + fname
    directory = dirname(dirname(dirname(abspath(__file__)))) + '/data/model/'
    pickle_filename = directory + netname + '.pickle'
    extracted_layers = None
    with open(pickle_filename, 'rb') as f:
        extracted_layers = pickle.load(f)
    return extracted_layers

def main():
    # model_name = '4c2d_noBN_4c2d_reg_1e_3_0'
    # input_lb = imageio.imread("../../data/cifar_lb.png") / 255.0
    # input_ub = imageio.imread("../../data/cifar_ub.png") / 255.0
    # print(np.max(input_ub), np.min(input_lb))

    # eps = 1.0/255.0
    eps = 1e-3
    # input_lb = np.clip(input_lb - eps, 0.0, 1.0)
    # input_ub = np.clip(input_ub + eps, 0.0, 1.0)
    # input_lb = np.zeros((28,28,1))
    # input_ub = np.ones((28,28,1))
    np.random.seed(12)
    sample = np.random.rand(28,28,1)
    input_lb = np.clip(sample - 0.005, 0.0, 1.0)
    input_ub = np.clip(sample + 0.005, 0.0, 1.0)
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    # Layers = load_extracted_model(model_name)
    Layers, model_name = load_model()
    test_NN = NeurNet(model_name, alpha=True)

    # NOTE: if needs the bounds of the last hidden layer, using Layers[:-1] to initialize the neural network, i.e., consider the last hidden layer as output layer. As shown in the following.
    # test_NN.setupNeuralNetwork(Layers[:-1], input_lb, input_ub, diff_lb, diff_ub)
    test_NN.setupNeuralNetwork(Layers, input_lb, input_ub, diff_lb, diff_ub)
    lb, ub, dlb, dub, dA_lb, dA_ub, bias_lb, bias_ub = test_NN.bound_backward()
    print('output bound:', lb.numpy(), ub.numpy())
    print('output variation bound:', dlb.numpy(), dub.numpy())
    print('matrix shape of linear bounds:', dA_lb.shape, dA_ub.shape)
    print('bias of linear bounds:', bias_lb.numpy(), bias_ub.numpy())

if __name__ == '__main__':
    main()

