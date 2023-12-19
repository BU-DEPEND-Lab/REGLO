from Layer_ITNE import Layer_ITNE
import tensorflow as tf
import  numpy as np

class Maxpool2d_ITNE(tf.keras.layers.MaxPool2D, Layer_ITNE):
    """
        Only support channel_last data format
        Currenly only support strides == pool_size
    """
    def __init__(self, input_shape = None, pool_size=(2, 2), padding='valid', **kwargs):
        super().__init__(pool_size, None, padding, data_format=None, **kwargs)
        Layer_ITNE.__init__(self)
        self.pool_size = pool_size
        self.n_splits = pool_size[0] * pool_size[1]
        # if strides is None:
        #     self.strides = self.pool_size
        # else:
        #     self.strides = strides
        self.strides = self.pool_size
        self.padding = padding
        self.my_input_shape = input_shape
        self.my_output_shape = None
        self._update_output_shape()

    def _spatial_shape(self, input_shape, pool, stride):
        if self.padding == 'valid':
            return ((input_shape - pool) // stride) + 1
        else:
            return ((input_shape - 1) // stride) + 1

    def _update_output_shape(self):
        if self.my_input_shape is None:
            self.my_output_shape = None
        else:
            self.my_output_shape = (
                self._spatial_shape(self.my_input_shape[0], self.pool_size[0], self.strides[0]),
                self._spatial_shape(self.my_input_shape[1], self.pool_size[1], self.strides[1]),
                self.my_input_shape[2]
            )
        # print(self.my_output_shape)
        
    def update_shape(self, input_shape):
        self.my_input_shape = input_shape
        self._update_output_shape()
    
    def backward(self, last_dA, inputs_bounds):
        """
            Currenly only support strides == pool_size
        """
        split_index = lambda i, j: i * self.pool_size[1] + j
        expand_dims = lambda A: tf.expand_dims(tf.expand_dims(A, axis=-3), axis=-2)
        if len(last_dA) < self.n_splits:
            zeros = None
            for i in last_dA:
                zeros = tf.zeros_like(last_dA[i])
                break
            for i in range(self.n_splits):
                if i not in last_dA:
                    last_dA[i] = zeros
        
        dA = tf.concat(
            [
                tf.concat(
                    [
                        expand_dims(last_dA[split_index(i, j)]) for j in range(self.pool_size[1])
                    ],
                    axis=-2
                ) for i in range(self.pool_size[0])
            ],
            axis=-4
        )
        shape = dA.shape
        shape = list(shape[:-5]) + [shape[-5]*shape[-4], shape[-3]*shape[-2], shape[-1]]
        dA = tf.reshape(dA, shape)

        shape = list(dA.shape)
        if self.padding == 'valid': # need to expand dA to the output shape
            padding = np.zeros((len(shape), 2), dtype=int)
            if shape[-3] < self.my_input_shape[-3]:
                padding[-3, 1] = self.my_input_shape[-3] - shape[-3]
            if shape[-2] < self.my_input_shape[-2]:
                padding[-2, 1] = self.my_input_shape[-2] - shape[-2]
            dA = tf.pad(dA, padding)
        else: # need to truncate dA to the ouput shape
            # Warning, may not be correct to directly abandon the padding part
            # as 'same' padding for maxpool pads -infy rather than 0
            start = tf.zeros_like(shape)
            if shape[-3] > self.my_input_shape[-3]:
                start[-3] = (shape[-3] - self.my_input_shape[-3]) // 2
                shape[-3] = start[-3] + self.my_input_shape[-3]
            if shape[-2] > self.my_input_shape[-2]:
                start[-2] = (shape[-2] - self.my_input_shape[-2]) // 2
                shape[-2] = start[-2] + self.my_input_shape[-2]
            dA = tf.slice(dA, start, shape) # can use tf.image.crop_to_bounding_box instead

        bias = 0
        return [dA], bias

    def eyes(self):
        print("ERROR: the output layer(s) of Maxpool2D should all be virtual layers, which don't need bounds of this Maxpool2D layer.")
        return None
        

