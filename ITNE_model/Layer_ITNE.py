import tensorflow as tf

class Layer_ITNE:
    def __init__(self):
        self.need_bounds = False
    
    def backward(self, last_dA, inputs_bounds):
        """
            Return:
                dAs: a list of dAs w.r.t. layer inputs
                bias
        """
        return [last_dA], 0
    
    def eyes(self):
        """
            Return the identical matrix with the size of the layer output
        """
        eye = tf.eye(1)
        return tf.stack([eye, -eye])

    def update_shape(self, input_shape):
        pass

    def update_shapes(self, input_shape, output_shape):
        pass