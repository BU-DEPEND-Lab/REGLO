import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras.models import model_from_json

from reglo import Reglo
import pickle

import time

def load_model(netname):
    json_filename = netname+".json"
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights(netname+".h5")
    return dnn_model

class Reglo_MNIST_PGD_EVAL(Reglo):
    def __init__(self, netname:str) -> None:
        tf_model = load_model(netname)
        # Prepare MNIST dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        # Specify the whole input space as the input region.
        input_lb = tf.zeros_like(x_train[0])
        input_ub = tf.ones_like(x_train[0])

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(500).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        super().__init__(tf_model, input_lb, input_ub, delta=5e-2, eps=1, area_search='milp', milp_max_regions=30, dataset=(train_ds, test_ds), model_name=netname.split('/')[-1])

    def eval_accuracy(self):
        """
        Evaluate the accuracy of self.model according to the testing data in self.dataset
        Result is appended to the end of self.results['accuracy']
        """
        _, test_ds = self.dataset
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        @tf.function
        def test_step(images, labels):
            predictions = self.model(images, training=False)
            test_accuracy(labels, predictions)

        test_accuracy.reset_states()
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        acc = test_accuracy.result().numpy()
        self.results['accuracy'].append(acc)
        return acc


if __name__ == '__main__':
    # netname = './adversarial_training/model/mnist_local_adv/20220811-154328/model_mnist_pgd_eps_0.3_0.2_steps_10_200_epoch_100_batch_256_lr_0.001'
    # netname = './adversarial_training/model/mnist_global_adv/20220811-162508/model_mnist_pgd_eps_0.3_0.2_steps_10_200_output_bound_60.0_loss_coeff_0.1_points_100_1000_epoch_100_batch_256_lr_0.001'
    netname = './adversarial_training/model/mnist_ST-AT-G_fine_tuning/20220814-022801/model_mnist_pgd_eps_0.3_0.2_steps_10_200_output_bound_60.0_loss_coeff_0.1_points_100_1000_epoch_100_batch_256_lr_0.0001'
    # netname = './adversarial_training/model/mnist_ST-AT-G_fine_tuning/20220814-022801/model_mnist_pgd_eps_0.3_0.2_steps_10_200_output_bound_60.0_loss_coeff_0.1_points_100_1000_epoch_100_batch_256_lr_0.0001'
    reglo_mnist = Reglo_MNIST_PGD_EVAL(netname)
    acc = reglo_mnist.eval_accuracy()
    print("Testing Accuracy", acc)
    reglo_mnist.eval_pgd(num_of_points=2000)
    reglo_mnist.eval_global_robustness()
    # t0 = time.time()
    # reglo_mnist.repair_one_step(repair_steps=150, repair_step_size=0.3)
    # print("repair time:", time.time() - t0)
    # acc = reglo_mnist.eval_accuracy()
    # print("Testing Accuracy", acc)
    # reglo_mnist.milp_worst_gradient()
    # reglo_mnist.eval_pgd(num_of_points=2000)
    # reglo_mnist.eval_global_robustness()


