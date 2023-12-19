import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import imageio

from reglo import Reglo
from mnist_class_cube import mnist_class_cube
from mnist_sample import load_model, build_dataset

class Reglo_MNIST(Reglo):
    def __init__(self, target_class=None) -> None:
        tf_model = load_model()
        train_ds, test_ds = build_dataset()
        if target_class is None: 
            input_lb = imageio.imread("../data/mnist_lb.png")
            input_ub = imageio.imread("../data/mnist_ub.png")
            input_lb = input_lb[..., tf.newaxis].astype("float32")
            input_ub = input_ub[..., tf.newaxis].astype("float32")
        else:
            input_lb, input_ub = mnist_class_cube(target_class=target_class)
            input_lb = input_lb[..., tf.newaxis].numpy()
            input_ub = input_ub[..., tf.newaxis].numpy()
        input_lb, input_ub = input_lb / 255.0, input_ub / 255.0
        super().__init__(tf_model, input_lb, input_ub, delta=2/255, eps=0.3, milp_max_regions=10, area_search='sample', sample_mode="dataset", dataset=(train_ds, test_ds), model_name='mnist_2c2d')


if __name__ == '__main__':
    for target_class in [9]:
        print(f'target_class: {target_class}.'+'-'*50)
        reglo_mnist = Reglo_MNIST(target_class)
        acc = reglo_mnist.eval_accuracy()
        print("Testing Accuracy", acc)
        reglo_mnist.eval_pgd(1000, 0.3)
        worst_gradient_in_sample = reglo_mnist.sample_worst_gradient()
        print(f"worst_gradient_in_sample_before_repair = {worst_gradient_in_sample}")
        
        reglo_mnist.repair_one_step(repair_steps=100, repair_step_size=0.1, t=1)
        
        acc = reglo_mnist.eval_accuracy()
        print("Testing Accuracy", acc)
        worst_gradient_in_sample = reglo_mnist.sample_worst_gradient()
        print(f"worst_gradient_in_sample_after_repair = {worst_gradient_in_sample}")
        reglo_mnist.eval_pgd(1000, 0.3)

