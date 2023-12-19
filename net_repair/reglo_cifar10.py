import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import imageio
from reglo import Reglo

def load_model():
    path = './adversarial_training/model/cifar10_clean/20221007-154526/'
    netname = 'model_cifar10_epoch_100_batch_256_lr_0.001'
    json_filename = path +netname+".json"
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights(path+netname+".h5")
    return dnn_model

def build_dataset():
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(256)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)
    return train_ds, test_ds

class Reglo_CIFAR10(Reglo):
    def __init__(self, target_class=None) -> None:
        tf_model = load_model()
        train_ds, test_ds = build_dataset()
        if target_class is None: 
            input_lb = imageio.imread("../data/cifar_lb.png")
            input_ub = imageio.imread("../data/cifar_ub.png")
            input_lb = input_lb.astype("float32")
            input_ub = input_ub.astype("float32")
        else:
            input_lb, input_ub = mnist_class_cube(target_class=target_class)
            input_lb = input_lb.numpy()
            input_ub = input_ub.numpy()
        input_lb, input_ub = input_lb / 255.0, input_ub / 255.0
        super().__init__(tf_model, input_lb, input_ub, delta=2/255, eps=10, milp_max_regions=10, area_search='sample', sample_mode="dataset", dataset=(train_ds, test_ds), model_name='cifar10')


if __name__ == '__main__':
    reglo_cifar10 = Reglo_CIFAR10()
    acc = reglo_cifar10.eval_accuracy()
    print("Testing Accuracy", acc)
    reglo_cifar10.eval_pgd(1000, 0.3)
    worst_gradient_in_sample = reglo_cifar10.sample_worst_gradient()
    print(f"worst_gradient_in_sample_before_repair = {worst_gradient_in_sample}")
    
    reglo_cifar10.repair_one_step(repair_steps=100, repair_step_size=0.1, t=1)
    
    acc = reglo_cifar10.eval_accuracy()
    print("Testing Accuracy", acc)
    worst_gradient_in_sample = reglo_cifar10.sample_worst_gradient()
    print(f"worst_gradient_in_sample_after_repair = {worst_gradient_in_sample}")
    reglo_cifar10.eval_pgd(1000, 0.3)