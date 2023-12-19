import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import argparse

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tensorflow_datasets as tfds

import imageio
import numpy as np

from reglo import Reglo

def parse_args():
    parser = argparse.ArgumentParser(
        description='German credit fairness experiment comparison.'
    )

    parser.add_argument(
        '--mode', type=str, choices=[
        'clean', 'local_adv', 'global_adv', 'AT-AT-G_fine_tuning', 'ST-AT-G_fine_tuning'
        ],
        help='The fashion of the trained model', required=True
    )

    parser.add_argument(
        '--model_name', type=str,
        help='name of the trained model', required=True,
    )

    parser.add_argument(
        '--age_range_lb', type=int, choices=[
        19, 25, 55, 65,
        ],
        help='The lower bound of age range.',
        required=True
    )

    parser.add_argument(
        '--age_range_ub', type=int, choices=[
        24, 54, 64, 75,
        ],
        help='The upper bound of age range,',
        required=True
    )

    return parser.parse_args()

def load_model(mode, netname):
    """
    load the trained model for german credit
    """
    if mode == 'clean':
        path = "./adversarial_training/model/german_credit_clean/"

    elif mode == 'local_adv':
        path = "./adversarial_training/model/german_credit_local_adv"

    elif mode == 'global_adv':
        path = './adversarial_training/model/german_credit_global_adv/'

    elif mode == 'ST-AT-G_fine_tuning':
        path = './adversarial_training/model/german_credit_ST-AT-G_fine_tuning/'

    elif mode == 'AT-AT-G_fine_tuning':
        path = './adversarial_training/model/german_credit_AT-AT-G_fine_tuning/'

    json_filename = path +netname+".json"
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights(path+netname+".h5")
    return dnn_model

def German_dataset(age_range_lb, age_range_ub):
    assert age_range_lb < age_range_ub

    MAX_AGE = 75
    MIN_AGE = 19
    AGE_INDEX = 12

    features = np.loadtxt(
        './adversarial_training/data/german_credit_redone.csv', delimiter=',',
        usecols=list(range(20))
    ).astype(np.float32)
    labels = np.loadtxt(
        './adversarial_training/data/german_credit_redone.csv', delimiter=',',
        usecols=-1
    ).astype(np.int)

    # Training and testing split ratio: 700/200
    x_train, x_test = features[:700].astype(np.float32), \
        features[700:].astype(np.float32)
    y_train, y_test = labels[:700], labels[700:]

    # Min and Max feature value.
    featstd = np.loadtxt(
        './adversarial_training/data/german_credit_featstd.csv', delimiter=',',
        usecols=list(range(20))).astype(np.float32)

    featmin_ = featstd[0, :]
    featmax_= featstd[1, :]
    featstd_ = featmax_ - featmin_

    # Normalize the features.
    x_train, x_test = (x_train - featmin_) / featstd_, \
        (x_test - featmin_) / featstd_

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(256)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

    # Assertion of age range
    assert MAX_AGE == featmax_[AGE_INDEX]
    assert MIN_AGE == featmin_[AGE_INDEX]
    assert age_range_lb >= featmin_[AGE_INDEX]

    # Construct lower bound and upper bound
    input_lb = np.zeros(shape=20)
    input_lb[AGE_INDEX] = (age_range_lb - MIN_AGE) / (MAX_AGE - MIN_AGE)
    input_lb = tf.convert_to_tensor(input_lb, np.float32)

    input_ub = np.ones(shape=20)
    input_ub[AGE_INDEX] = (age_range_ub - MIN_AGE) / (MAX_AGE - MIN_AGE)
    input_ub = tf.convert_to_tensor(input_ub, np.float32)

    delta = (age_range_ub - age_range_lb) / (MAX_AGE - MIN_AGE)

    return train_ds, test_ds, input_lb, input_ub, delta, AGE_INDEX

class Reglo_German(Reglo):
    def __init__(self, mode, model_name, age_range_lb, age_range_ub) -> None:
        tf_model = load_model(mode, model_name)
        train_ds, test_ds, input_lb, input_ub, delta, AGE_INDEX = German_dataset(age_range_lb, age_range_ub)
        #eps = 0.01 for all age bound
        super().__init__(tf_model, input_lb, input_ub, delta=delta, eps=0.01, milp_max_regions=30, area_search='sample', sample_mode="random", dataset=(train_ds, test_ds), model_name='german_credit', fairness_index=AGE_INDEX)

    def eval_accuracy(self):
        """
        Evaluate the accuracy of self.model according to the testing data in self.dataset
        Result is appended to the end of self.results['accuracy']
        Binary classification should have sigmoid activation for the output layer
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

    AGE_INDEX = 12

    args = parse_args()

    reglo_german = Reglo_German(args.mode, args.model_name, args.age_range_lb, args.age_range_ub)
    acc = reglo_german.eval_accuracy()
    print("Testing Accuracy", acc)
    reglo_german.eval_pgd(num_of_points=1000, out_bound=0.05, single_idx=AGE_INDEX)

    reglo_german.eval_global_robustness()

    # worst_gradient_in_sample = reglo_german.sample_worst_gradient()
    # print(f"before_repair_worst_gradient_in_sample = {worst_gradient_in_sample}")
    # reglo_german.repair_one_step(repair_steps=100, repair_step_size=0.5)
    # acc = reglo_german.eval_accuracy()
    # print("Testing Accuracy", acc)
    # worst_gradient_in_sample = reglo_german.sample_worst_gradient()
    # print(f"after_repair_worst_gradient_in_sample = {worst_gradient_in_sample}")
    # reglo_german.eval_pgd(num_of_points=1000, out_bound=0.05, single_idx=AGE_INDEX)

    # reglo_german.eval_global_robustness()
    print()
    print()
