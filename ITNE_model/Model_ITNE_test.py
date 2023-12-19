from Flatten_layer_ITNE import Flatten_ITNE
from Model_ITNE import Model_ITNE
from Linear_layer_ITNE import Linear_ITNE
from Conv2d_layer_ITNE import Conv2d_ITNE
from Relu_layer_ITNE import Relu_ITNE
import tensorflow as tf
import numpy as np

class Test_Model_ITNE:
    def test_inAndOutBounds(self):
        #        input
        #        /  |
        #       0   |
        #     / | \ |
        #    2  3   1
        #    \ /    |
        #     5     4
        #       \ /
        #        6
        model = Model_ITNE()
        model.graph[0] = [-1]
        model.graph[1] = [-1, 0]
        model.graph[2] = [0]
        model.graph[3] = [0]
        model.graph[4] = [1]
        model.graph[5] = [2,3]
        model.graph[6] = [4,5]
        
        # Test inbounds
        inbounds = model.get_inbounds(model.graph)
        assert inbounds[5] == 2
        assert inbounds[0] == 1
        assert inbounds[1] == 2


        # Test outbounds
        model.update_outbounds()
        assert len(model.outbounds[0]) == 3
        assert len(model.outbounds[-1]) == 2
        assert len(model.outbounds[3]) == 1

        outbounds = model.get_outbounds(model.graph, 5)
        assert outbounds[-1] == 1
        assert outbounds[0] == 2
        outbounds = model.get_outbounds(model.graph, 3)
        assert outbounds[0] == 1
        outbounds = model.get_outbounds(model.graph, 6)
        assert outbounds[-1] == 2
        assert outbounds[0] == 3
        assert outbounds[1] == 1

    def test_linear_layer(self):
        model = Model_ITNE()
        layer_id = model.add_layer(Linear_ITNE(20))
        layer_id = model.add_layer(Linear_ITNE(2), input_ids=[layer_id])
        model.update_outbounds()
        input = tf.random.uniform(shape=[16, 10])
        output = model(input, training=False)
        assert len(output.shape) == 2
        assert output.shape == (16, 2)

        input_dist_lb = -0.1*np.ones(10)
        input_dist_ub = 0.1*np.ones(10)
        model.set_input_bounds(input_dist_lb, input_dist_ub)
        res = model.output_variation()
        assert res.shape == (2,)

    def test_relu_itne(self):
        input = tf.random.uniform(shape=[16, 2, 3])
        relu = Relu_ITNE((2,3))
        output = relu(input)
        print(output.shape)

        # Test eyes() function for multi-dim shape
        eyes = relu.eyes()
        assert eyes.shape == (2, 2, 3, 2, 3)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    for l in range(3):
                        if i == k and j == l:
                            assert eyes[0,i,j,k,l] == 1
                            assert eyes[1,i,j,k,l] == -1
                        else:
                            assert eyes[0,i,j,k,l] == 0
                            assert eyes[1,i,j,k,l] == 0

        # Test backward() function for multi-dim shape
        dA = tf.random.uniform((2,5,2,3))
        bounds = tf.stack([-tf.random.uniform((2,3)), -tf.random.uniform((2,3))])
        dAs, bias = relu.backward(dA, [bounds])
        assert dAs[0].shape == (2,5,2,3)
        assert bias.shape == (2,5)

        dA = tf.random.uniform((2,4,4,1,2,3))
        bounds = tf.stack([-tf.random.uniform((2,3)), -tf.random.uniform((2,3))])
        dAs, bias = relu.backward(dA, [bounds])
        assert dAs[0].shape == (2,4,4,1,2,3)
        assert bias.shape == (2,4,4,1)

    def test_conv2d(self):
        input = tf.random.uniform(shape=[16, 5, 5, 3])
        conv = Conv2d_ITNE(5, 3)
        output = conv(input)
        print(output.shape)
        conv.update_shapes(input.shape[1:], output.shape[1:])
        eyes = conv.eyes()
        assert eyes.shape == (2, 3, 3, 5, 3, 3, 5)

        # Test backward() function for multi-dim shape
        dA = tf.random.uniform((2,8,3,3,5))
        dAs, _ = conv.backward(dA, None)
        assert dAs[0].shape == (2,8,5,5,3)

        dA = tf.random.uniform((2,3,4,6,3,3,5))
        dAs, _ = conv.backward(dA, None)
        assert dAs[0].shape == (2,3,4,6,5,5,3)

    def test_conv2d_stride(self):
        input = tf.random.uniform(shape=[16, 7, 7, 3])
        conv = Conv2d_ITNE(5, 3, strides=2)
        output = conv(input)
        print(output.shape)
        conv.update_shapes(input.shape[1:], output.shape[1:])
        eyes = conv.eyes()
        assert eyes.shape == (2, 3, 3, 5, 3, 3, 5)

        # Test backward() function for multi-dim shape
        dA = tf.random.uniform((2,8,3,3,5))
        dAs, _ = conv.backward(dA, None)
        assert dAs[0].shape == (2,8,7,7,3)

        dA = tf.random.uniform((2,3,4,6,3,3,5))
        dAs, _ = conv.backward(dA, None)
        assert dAs[0].shape == (2,3,4,6,7,7,3)

    def test_flatten(self):
        input = tf.random.uniform(shape=[16, 5, 5, 3])
        flatten = Flatten_ITNE()
        output = flatten(input)
        flatten.update_shape(input.shape[1:])
        
        # test eyes function
        eyes = flatten.eyes()
        assert eyes.shape == (2, 75, 75)

        # test backward function
        dA = tf.random.uniform((2,8,3,3,75))
        dAs, _ = flatten.backward(dA, None)
        assert dAs[0].shape == (2,8,3,3,5,5,3)

