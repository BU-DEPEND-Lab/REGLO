import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from enum import Enum, auto
from collections import deque
import tensorflow as tf
import numpy as np

from Layer_ITNE import Layer_ITNE
from Maxpool2d_layer_ITNE import Maxpool2d_ITNE
from Virtual_layer_ITNE import Virtual_ITNE
from Subtract_layer_ITNE import Subtract_ITNE
from Add_layer_ITNE import Add_ITNE
from Relu_layer_ITNE import Relu_ITNE
from Conv2d_layer_ITNE import Conv2d_ITNE

class Model_ITNE(tf.keras.Model):
    def __init__(self):
        super(Model_ITNE, self).__init__()
        self.graph = {}     # -1 represent the input layer
        self.forward_graph = {}     # Does not contain pseudo layers used in backward
        self.n_layers = 0
        self.myLayers = []
        self.output_layer = -1
        self.forward_output_layer = -1
        self.outbounds = {}
        self.forward_outbounds = {}
        self.input_dist_lb = None
        self.input_dist_ub = None
        self.input_dim = 1
        self.maxpool2d_layers = {}  # format: {maxpool_layer_id: last_pseudo_layer_id}.
        self.maxpool2d_indices = {} # format: {maxpool_layer_id: post_layer_mapping}.
                                    # post_layer_mapping format: {post_layer_id: index in maxpool}.
                                    # Explanation: A nxn maxpooling layer will split the input into n^2 
                                    # subtensors, index from 0 to (n^2 - 1).

    def add_layer(self, layer, input_ids = [-1], isOutput=True):
        if not isinstance(layer, Layer_ITNE):
            print("[Add Layer FAILED] Unsupported ITNE layer type.")
            return -1
        if isinstance(layer, Maxpool2d_ITNE):
            print("[Add Layer FAILED] Please use add_maxpool_layer() method to add Maxpool2D layers.")
            return -1
        layer_id = self.n_layers
        self.n_layers += 1
        self.myLayers.append(layer)
        backward_inputs = [(self.maxpool2d_layers[i] if i in self.maxpool2d_layers else i) for i in input_ids]
        self.graph[layer_id] = backward_inputs
        self.forward_graph[layer_id] = input_ids
        if isOutput:
            self.output_layer = layer_id
            self.forward_output_layer = layer_id
        return layer_id

    def add_maxpool_layer(self, input_id = -1, pool_size=(2, 2), strides=None, padding='valid', isOutput=True):
        if padding != 'valid':
            print("Warning: ITNE maxpooling with 'same' padding has not validate to be correct.")
        layer = Maxpool2d_ITNE(pool_size=pool_size, padding=padding)
        maxpool_layer_id = self.n_layers
        self.n_layers += 1
        self.myLayers.append(layer)
        self.graph[maxpool_layer_id] = [input_id]
        self.forward_graph[maxpool_layer_id] = [input_id]
        self.maxpool2d_indices[maxpool_layer_id] = {}
        remain_layers = deque()
        for i in range(layer.n_splits):
            virtual_layer = Virtual_ITNE()
            virtual_layer_id = self.n_layers
            self.n_layers += 1
            self.myLayers.append(virtual_layer)
            self.graph[virtual_layer_id] = [maxpool_layer_id]
            self.maxpool2d_indices[maxpool_layer_id][virtual_layer_id] = i
            remain_layers.append(virtual_layer_id)
        while len(remain_layers) > 1:
            l1 = remain_layers.popleft()
            l2 = remain_layers.popleft()
            remain_layers.append(self.add_max_block(l1, l2))
        output_layer_id = remain_layers.pop()
        self.maxpool2d_layers[maxpool_layer_id] = output_layer_id
        if isOutput:
            self.output_layer = output_layer_id
            self.forward_output_layer = maxpool_layer_id
        return maxpool_layer_id

    def add_max_block(self, input_layer1_id, input_layer2_id):
        # max(a, b) = a + relu(b - a)
        sub_layer = Subtract_ITNE()
        sub_layer_id = self.n_layers
        self.n_layers += 1
        self.myLayers.append(sub_layer)
        self.graph[sub_layer_id] = [input_layer2_id, input_layer1_id]
        relu_layer = Relu_ITNE()
        relu_layer_id = self.n_layers
        self.n_layers += 1
        self.myLayers.append(relu_layer)
        self.graph[relu_layer_id] = [sub_layer_id]
        add_layer = Add_ITNE()
        add_layer_id = self.n_layers
        self.n_layers += 1
        self.myLayers.append(add_layer)
        self.graph[add_layer_id] = [input_layer1_id, relu_layer_id]
        return add_layer_id

    def set_input_bounds(self, input_dist_lb, input_dist_ub):
        self.input_dist_lb = tf.constant(input_dist_lb, dtype=tf.float32)
        self.input_dist_ub = tf.constant(input_dist_ub, dtype=tf.float32)
        self.input_dim = len(self.input_dist_lb.shape)

    def update_outbounds(self):
        res = {}
        for i in self.graph:
            for j in self.graph[i]:
                if j in res:
                    res[j].append(i)
                else:
                    res[j] = [i]
        self.outbounds = res
        res = {}
        for i in self.forward_graph:
            for j in self.forward_graph[i]:
                if j in res:
                    res[j].append(i)
                else:
                    res[j] = [i]
        self.forward_outbounds = res 

    def get_inbounds(self, graph):
        res = {i: len(graph[i]) for i in graph}
        return res

    def get_outbounds(self, graph, out_layer_id):
        related_layers = set()
        q = [out_layer_id]
        while q:
            q_new = []
            for out_id in q:
                for in_id in graph[out_id]:
                    if in_id not in related_layers:
                        related_layers.add(in_id)
                        if in_id >= 0:
                            q_new.append(in_id)
            q = q_new
        outbounds = {}
        for i in related_layers:
            outbounds[i] = 0
        related_layers.add(out_layer_id)
        for out_id in related_layers:
            if out_id < 0:
                continue
            for in_id in graph[out_id]:
                outbounds[in_id] += 1

        return outbounds

    def call(self, inputs):
        outputs = {-1: inputs}
        inbounds = self.get_inbounds(self.forward_graph)
        ready = deque()
        for i in self.forward_outbounds[-1]:
            inbounds[i] -= 1
            if inbounds[i] == 0:
                ready.append(i)
        while ready:
            layer_id = ready.pop()
            layer = self.myLayers[layer_id]
            if len(self.forward_graph[layer_id]) == 1:
                layer_in = outputs[self.forward_graph[layer_id][0]]
            else:
                layer_in = [outputs[i] for i in self.forward_graph[layer_id]]
            outputs[layer_id] = layer(layer_in)
            if layer_id in self.forward_outbounds:
                for i in self.forward_outbounds[layer_id]:
                    inbounds[i] -= 1
                    if inbounds[i] == 0:
                        ready.append(i)
                    
        return outputs[self.forward_output_layer]

    def _post_layers_need_my_bound(self, layer_id):
        if layer_id in self.outbounds:
            for post_layer_id in self.outbounds[layer_id]:
                post_layer = self.myLayers[post_layer_id]
                if post_layer.need_bounds:
                    return True
        return False


    def output_variation(self):
        # bounds = {-1: input_diff_bounds}
        bounds = {}     # we do not store the input lower and upper bounds in bounds
                        # for each layer, the bounds include the lower bound and negative upper bound
        inbounds = self.get_inbounds(self.graph)
        ready = deque()
        for i in self.outbounds[-1]:
            inbounds[i] -= 1
            if inbounds[i] == 0:
                ready.append(i)
        while ready:
            layer_id = ready.pop()
            layer = self.myLayers[layer_id]
            if self._post_layers_need_my_bound(layer_id) or layer_id == self.output_layer:
                bounds[layer_id] = self.output_variation_layer(layer_id, bounds)
            if layer_id in self.outbounds:
                for i in self.outbounds[layer_id]:
                    inbounds[i] -= 1
                    if inbounds[i] == 0:
                        ready.append(i)
        lbs = bounds[self.output_layer][0]
        neg_ubs = bounds[self.output_layer][1]
        return - (neg_ubs + lbs)    # i.e., ubs - lbs

    def output_variation_layer(self, out_layer_id, bounds):
        out_layer = self.myLayers[out_layer_id]
        dAs = {out_layer_id: out_layer.eyes()}
        outbounds = self.get_outbounds(self.graph, out_layer_id)
        ready = deque()
        ready.append(out_layer_id)
        bias = 0
        while ready:
            layer_id = ready.pop()
            layer = self.myLayers[layer_id]
            needed_bounds = None
            if layer.need_bounds:
                needed_bounds = []
                for in_id in self.graph[layer_id]:
                    needed_bounds.append(bounds[in_id])
            in_dAs, out_bias = layer.backward(dAs[layer_id], needed_bounds)
            # print(layer_id, dAs[layer_id].shape, in_dAs[0].shape)
            bias += out_bias
            for dA, in_id in zip(in_dAs, self.graph[layer_id]):
                if in_id in self.maxpool2d_indices:
                    pool_index = self.maxpool2d_indices[in_id][layer_id]
                    if in_id in dAs:
                        dAs[in_id][pool_index] = dA
                    else:
                        dAs[in_id] = {pool_index: dA}
                else:
                    if in_id in dAs:
                        dAs[in_id] = dAs[in_id] + dA
                    else:
                        dAs[in_id] = dA
                outbounds[in_id] -= 1
                if in_id >= 0 and outbounds[in_id] == 0:
                    ready.append(in_id)
        # print(dAs[-1].shape)
        out_dist_bound = (      # lower bound and negative upper bound
            tf.tensordot(tf.math.maximum(dAs[-1], 0), self.input_dist_lb, axes=self.input_dim) + 
            tf.tensordot(tf.math.minimum(dAs[-1], 0),  self.input_dist_ub, axes=self.input_dim) + 
            bias)
        return out_dist_bound

    def update_shapes(self, input_shape):
        batch_size = 1
        inputs = tf.random.uniform([batch_size]+list(input_shape))
        outputs = {-1: inputs}
        inbounds = self.get_inbounds(self.graph)
        ready = deque()
        for i in self.outbounds[-1]:
            inbounds[i] -= 1
            if inbounds[i] == 0:
                ready.append(i)
        while ready:
            layer_id = ready.pop()
            layer = self.myLayers[layer_id]
            in_shape = outputs[self.graph[layer_id][0]].shape[1:]
            if len(self.graph[layer_id]) == 1:
                layer_in = outputs[self.graph[layer_id][0]]
            else:
                layer_in = [outputs[i] for i in self.graph[layer_id]]
            outputs[layer_id] = layer(layer_in)
            out_shape = outputs[layer_id].shape[1:]
            # print(layer_id, type(layer).__name__, "was run. Shapes:", in_shape, out_shape)
            if isinstance(layer, Conv2d_ITNE):
                layer.update_shapes(in_shape, out_shape)
            else:
                layer.update_shape(in_shape)
            if layer_id in self.outbounds:
                for i in self.outbounds[layer_id]:
                    inbounds[i] -= 1
                    if inbounds[i] == 0:
                        ready.append(i)





        


        
