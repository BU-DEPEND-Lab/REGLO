import os
import argparse
import time
import datetime
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import model_from_json

from PGD import ProjectedGradientDescent, ProjectedGradientDescentForRegion

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='PGD evaluation for the global robustness.'
    )
    parser.add_argument(
        '--gpu_id', type=str, default='0', help='GPU ID for training',
        required=False
    )
    parser.add_argument(
        '--dataset', type=str, choices=[
            'mnist',
            'cryptojacker', 'twitter_spam_account', 'twitter_spam_url',
            'german_credit_numeric', 'german_credit', 'auto_mpg', 'cifar10'
        ],
        help='Dataset name.', required=True
    )
    parser.add_argument(
        '--training_mode', type=str,
        choices=['clean', 'local_adv', 'global_adv', 'ST-AT-G', 'AT-AT-G'],
        help='The mode for training.', required=True
    )
    parser.add_argument(
        '--eps_train', type=float, default=0.4,
        help='The adversarial perturbation bound for training.',
        required=False
    )
    parser.add_argument(
        '--eps_test', type=float, default=0.3,
        help='The adversarial perturbation bound for testing.',
        required=False
    )
    parser.add_argument(
        '--eps_step', type=float, default=0.1,
        help='Attack step size (input variation) at each iteration.',
        required=False
    )
    parser.add_argument(
        '--PGD_steps_train', type=int, default=10,
        help='The number of steps for PGD attack at the training time.',
        required=False
    )
    parser.add_argument(
        '--PGD_steps_test', type=int, default=200,
        help='The number of steps for PGD attack at the testing time.',
        required=False
    )
    parser.add_argument(
        '--epoch', type=int, default=100,
        help='The number of training epochs.', required=False
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='The size of training batch size.', required=False
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='The learning rate for the optimizer.', required=False
    )
    parser.add_argument(
        '--global_robustness_training_num_of_points', type=int, default=100,
        help='The number of sampling points for adversarial global robustness '
        'training',
        required=False
    )
    parser.add_argument(
        '--global_robustness_test_num_of_points', type=int, default=1000,
        help='The number of sampling points for adversarial global robustness '
        'testing',
        required=False
    )
    parser.add_argument(
        '--global_robustness_coeff', type=float, default=0.1,
        help='The coefficient of the global robusntess objective.',
        required=False
    )
    parser.add_argument(
        '--global_robustness_output_bound', type=float, default=0.3,
        help='The output bound of the global robustness property.',
        required=False
    )
    parser.add_argument(
        '--preload_netname', type=str, default=None,
        help='The model name that needs to be fine tuned.',
        required=False
    )
    parser.add_argument(
        '--only_final_evaluation', action='store_true',
        help='Flag for whether only perform evaluation at the end.',
        required=False
    )
    parser.add_argument(
        '--target_class', type=int, default=None,
        help='A target class for the global robustness input region. Only support mnist now.',
        required=False
    )

    parser.add_argument(    
        '--age_range_lb', type=int, default=19, choices=[   
        19, 25, 55, 65, 
        ],  
        help='The lower bound of age range.',   
        required=False  
    )   
    parser.add_argument(    
        '--age_range_ub', type=int, default=75, choices=[   
        24, 54, 64, 75, 
        ],  
        help='The upper bound of age range,',   
        required=False  
    )

    return parser.parse_args()


def load_model(model_path_and_netname):
    """
    Load the model from the given path and netname.
    """
    json_filename = model_path_and_netname + ".json"
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights(model_path_and_netname + ".h5")

    return dnn_model


class PGD_train(object):
    """
    Train model with PGD perturbed inputs and test with PGD perturbed inputs.
    """
    def __init__(
        self,
        model,
        dataset,
        x_train,
        y_train,
        x_test,
        y_test,
        eps_train=0.4,
        eps_test=0.3,
        eps_step=0.1,
        PGD_steps_train=10,
        PGD_steps_test=200,
        epoch=100,
        batch_size=256,
        lr=0.0001,
        binary=False,
        training_mode='local_adv',
        input_lb=None,
        input_ub=None,
        global_robustness_training_num_of_points=100,
        global_robustness_test_num_of_points=100,
        global_robustness_coeff=0.1,
        global_robustness_output_bound=0.3,
        regresssion=False,
        fine_tune=False,
        target_class=None,
        single_idx=None,
    ):
        # The dataset and the model
        self.dataset = dataset
        self.model = model
        self.binary = binary
        self.single_idx = single_idx
        self.regression = regresssion
        self.target_class = target_class
        if self.regression:
            self.binary = False

        # Training mode: clean, local_adv, global_adv
        self.training_mode = training_mode

        # Prepare training and testing data
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(y_test.shape[0])

        # Prepare training objective
        if self.regression:
            self.loss_object = tf.keras.losses.MeanSquaredError()
        elif self.binary:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )
        else:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )

        # Prepare objective for the global robustness
        self.global_robustness_object = tf.keras.losses.MeanSquaredError()
        # The coefficient for global robustness loss.
        self.global_robustness_coeff = global_robustness_coeff
        # The output bound of the global robusntess property.
        self.global_robustness_output_bound = global_robustness_output_bound

        # Prepare optimizer
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Specify training loss and accuracy metric
        if self.regression:
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.MeanAbsoluteError(
                name='train_accuracy'
            )

            # Specify testing loss and accuracy metric
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.MeanAbsoluteError(
                name='test_accuracy'
            )
            self.PGD_test_loss = tf.keras.metrics.Mean(name='PGD_test_loss')
            self.PGD_test_accuracy = tf.keras.metrics.MeanAbsoluteError(
                name='PGD_test_accuracy'
            )
        elif self.binary:
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(
                name='train_accuracy'
            )

            # Specify testing loss and accuracy metric
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.BinaryAccuracy(
                name='test_accuracy'
            )
            self.PGD_test_loss = tf.keras.metrics.Mean(name='PGD_test_loss')
            self.PGD_test_accuracy = tf.keras.metrics.BinaryAccuracy(
                name='PGD_test_accuracy'
            )
        else:
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='train_accuracy'
            )

            # Specify testing loss and accuracy metric
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='test_accuracy'
            )
            self.PGD_test_loss = tf.keras.metrics.Mean(name='PGD_test_loss')
            self.PGD_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='PGD_test_accuracy'
            )

        # Specify the number of training epochs.
        self.epoch = epoch

        # Specify the size of a batch
        self.batch_size = batch_size

        # Specify the PGD attacker for training and testing
        self.pgd_train = ProjectedGradientDescent(
            self.model, eps=eps_train, eps_step=eps_step, steps=PGD_steps_train
        )
        self.pgd_test = ProjectedGradientDescent(
            self.model, eps=eps_test, eps_step=eps_step, steps=PGD_steps_test
        )

        if input_lb is not None and input_ub is not None:
            # Specify the PGD attacker to train and test the model for global
            # robustness property
            self.input_lb = input_lb
            self.input_ub = input_ub

            self.global_pgd_train = ProjectedGradientDescentForRegion(
                self.model, self.input_lb, self.input_ub, eps=eps_train,
                eps_step=eps_step, steps=PGD_steps_train
            )
            self.global_pgd_test = ProjectedGradientDescentForRegion(
                self.model, self.input_lb, self.input_ub, eps=eps_test,
                eps_step=eps_step, steps=PGD_steps_test
            )

            self.global_robustness_train_loss = tf.keras.metrics.Mean(
                name='global_robustness_train_loss'
            )
            self.global_robustness_train_violation_rate = \
                tf.keras.metrics.BinaryAccuracy(
                    name='global_robustness_train_violation_rate'
                )

            self.global_robustness_test_loss = tf.keras.metrics.Mean(
                name='global_robustness_train_loss'
            )
            self.global_robustness_test_violation_rate = \
                tf.keras.metrics.BinaryAccuracy(
                    name='global_robustness_test_violation_rate'
                )

        # The number of randomly sampled points in the input region.
        self.global_robustness_training_num_of_points = \
            global_robustness_training_num_of_points
        self.global_robustness_test_num_of_points = \
            global_robustness_test_num_of_points

        # Setup logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if not fine_tune:
            self.base_dir = './model/' + self.dataset + '_' + training_mode \
                + '/' + current_time + '/'
        else:
            self.base_dir = './model/' + self.dataset + '_' + training_mode \
                + '_' + 'fine_tuning/' + current_time + '/'
        train_log_dir = self.base_dir + 'train'
        test_log_dir = self.base_dir + 'test'

        self.train_summary_writer = tf.summary.create_file_writer(
            train_log_dir
        )
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x, training=True)
            loss = self.loss_object(y, predictions)

            if self.training_mode == 'global_adv':
                random_x, perturbed_x = self.global_pgd_train.random_sample(
                    self.global_robustness_training_num_of_points, single_idx=self.single_idx
                )
                global_robustness_loss = self.global_robustness_object(
                    self.model(random_x), self.model(perturbed_x)
                )
                output_diff = self.model(random_x) - self.model(perturbed_x)
                loss += self.global_robustness_coeff * global_robustness_loss

        # Compute the gradient and update the model with the gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        self.train_loss(loss)
        self.train_accuracy(
            y,
            predictions
            if not self.binary else tf.keras.activations.sigmoid(predictions)
        )
        if self.training_mode == 'global_adv':
            self.global_robustness_train_loss(global_robustness_loss)
            self.global_robustness_train_violation_rate(
                tf.ones_like(output_diff)
                # For multi-class classifiers
                if args.dataset not in ['mnist', 'german_credit', 'cifar10'] else tf.ones(output_diff.shape[0]),
                tf.cast(
                    tf.math.greater(
                        tf.norm(output_diff, ord=np.inf, axis=-1),
                        self.global_robustness_output_bound
                    ), tf.float32)
            )

    def test_step(self, x, y, perturbed_x=None):
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(x, training=False)
        test_loss = self.loss_object(y, predictions)

        if self.training_mode in ['global_adv', 'ST-AT-G', 'AT-AT-G']:
            random_x, perturbed_random_x = self.global_pgd_test.random_sample(
                self.global_robustness_test_num_of_points
            )
            global_robustness_loss = self.global_robustness_object(
                self.model(random_x), self.model(perturbed_random_x)
            )
            output_diff = self.model(random_x) - self.model(perturbed_random_x)

        # Compute test loss and accuracy
        self.test_loss(test_loss)
        self.test_accuracy(
            y,
            predictions
            if not self.binary else tf.keras.activations.sigmoid(predictions)
        )

        if perturbed_x is not None:
            # Test with PGD perturbed inputs when they are provided.
            perturbed_predictions = self.model(perturbed_x, training=False)
            PGD_test_loss = self.loss_object(y, perturbed_predictions)

            # Compute PGD test loss and accuracy
            self.PGD_test_loss(PGD_test_loss)
            self.PGD_test_accuracy(
                y,
                perturbed_predictions
                if not self.binary else tf.keras.activations.sigmoid(
                    perturbed_predictions
                )
            )

        if self.training_mode in ['global_adv', 'ST-AT-G', 'AT-AT-G']:
            self.global_robustness_test_loss(global_robustness_loss)
            self.global_robustness_test_violation_rate(
                tf.ones_like(output_diff)
                # For multi-class classifiers
                if args.dataset not in ['mnist', 'german_credit', 'cifar10'] else tf.ones(output_diff.shape[0]),

                tf.cast(
                    tf.math.greater(
                        tf.norm(output_diff, ord=np.inf, axis=-1),
                        self.global_robustness_output_bound
                    ), tf.float32)
            )

    def AT_G_fine_tune_step(self):
        with tf.GradientTape() as tape:
            random_x, perturbed_x = self.global_pgd_train.random_sample(
                self.global_robustness_training_num_of_points
            )
            global_robustness_loss = self.global_robustness_object(
                self.model(random_x), self.model(perturbed_x)
            )
            output_diff = self.model(random_x) - self.model(perturbed_x)
            loss = self.global_robustness_coeff * global_robustness_loss

        # Compute the gradient and update the model with the gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        self.global_robustness_train_loss(global_robustness_loss)
        self.global_robustness_train_violation_rate(
            tf.ones_like(output_diff)
            # For multi-class classifiers
            if args.dataset not in ['mnist', 'german_credit', 'cifar10'] else tf.ones(output_diff.shape[0]),
            tf.cast(
                tf.math.greater(
                    tf.norm(output_diff, ord=np.inf, axis=-1),
                    self.global_robustness_output_bound
                ), tf.float32)
        )

    def train(self, only_final_evaluation=False):
        """
        Train with clean training data - normal training.
        """
        for _epoch in range(self.epoch):
            t = time.time()
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for x, y in self.train_ds:
                self.train_step(x, y)

            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    'Training Loss', self.train_loss.result(), step=_epoch
                )
                tf.summary.scalar(
                    'Training Accuracy', self.train_accuracy.result(),
                    step=_epoch
                )

            if not only_final_evaluation or _epoch == self.epoch - 1:
                for x, y in self.test_ds:
                    self.test_step(x, y)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar(
                        'Test Loss', self.test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'Test Accuracy', self.test_accuracy.result(), step=_epoch
                    )

            print(
                f'Epoch {_epoch + 1}, '
                f'Loss: {self.train_loss.result(): .3f}, '
                f'Accuracy: {self.train_accuracy.result() * 100: .2f}, '
                f'Test Loss: {self.test_loss.result(): .3f}, '
                f'Test Accuracy: {self.test_accuracy.result() * 100: .2f}, '
                f'Training time: {time.time() - t: .2f}s',
                flush=True
            )
        file_name = "epoch_{}_batch_{}_lr_{}"\
            .format(self.epoch, self.batch_size, self.lr)
        # Save the trained model
        self.save_model(file_name)

    def train_pgd(self, only_final_evaluation=False):
        """
        Train with local adversarial examples.
        """
        for _epoch in range(self.epoch):
            t = time.time()
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for x, y in self.train_ds:
                # Train with PGD perturbed inputs.
                perturbed_x = self.pgd_train.generate(x, log=False)
                self.train_step(perturbed_x, y)

            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    'PGD Training Loss', self.train_loss.result(), step=_epoch
                )
                tf.summary.scalar(
                    'PGD Training Accuracy', self.train_accuracy.result(),
                    step=_epoch
                )

            if not only_final_evaluation or _epoch == self.epoch - 1:
                for x, y in self.test_ds:
                    # Test with PGD perturbed inputs.
                    perturbed_x = self.pgd_test.generate(x, log=False)
                    self.test_step(x, y, perturbed_x=perturbed_x)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar(
                        'Clean Test Loss', self.test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'Clean Test Accuracy', self.test_accuracy.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'PGD Test Loss', self.PGD_test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'PGD Test Accuracy', self.PGD_test_accuracy.result(),
                        step=_epoch
                    )

            print(
                f'Epoch {_epoch + 1}, '
                f'PGD Training Loss: {self.train_loss.result(): .3f}, '
                f'PGD Training Accuracy: {self.train_accuracy.result() * 100: .2f}, '
                f'Clean Test Loss: {self.test_loss.result(): .3f}, '
                f'Clean Test Accuracy: {self.test_accuracy.result() * 100: .2f}, '
                f'PGD Test Loss: {self.PGD_test_loss.result(): .3f}, '
                f'PGD Test Accuracy: {self.PGD_test_accuracy.result() * 100: .2f}, '
                f'Training time: {time.time() - t: .2f}s',
                flush=True
            )

        file_name = "pgd_eps_{}_{}_steps_{}_{}_epoch_{}_batch_{}_lr_{}"\
            .format(self.pgd_train.eps, self.pgd_test.eps,
                    self.pgd_train.steps, self.pgd_test.steps, self.epoch,
                    self.batch_size, self.lr)
        # Save the trained model
        self.save_model(file_name)

    def train_global_pgd(self, only_final_evaluation=False):
        """
        Adversarial training for the global robustness property.
        """
        for _epoch in range(self.epoch):
            t = time.time()
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.global_robustness_train_loss.reset_states()
            self.global_robustness_test_loss.reset_states()
            self.global_robustness_train_violation_rate.reset_states()
            self.global_robustness_test_violation_rate.reset_states()

            for x, y in self.train_ds:
                self.train_step(x, y)

            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    'Training Loss', self.train_loss.result(), step=_epoch
                )
                tf.summary.scalar(
                    'Training Global Robustness Loss',
                    self.global_robustness_train_loss.result(),
                    step=_epoch
                )
                tf.summary.scalar(
                    'Training Accuracy', self.train_accuracy.result(),
                    step=_epoch
                )
                tf.summary.scalar(
                    'Training Global Robustness Violation Rate',
                    self.global_robustness_train_violation_rate.result(),
                    step=_epoch
                )

            if not only_final_evaluation or _epoch == self.epoch - 1:
                for x, y in self.test_ds:
                    # Test with PGD perturbed inputs.
                    perturbed_x = self.pgd_test.generate(x, log=False)
                    self.test_step(x, y, perturbed_x=perturbed_x)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar(
                        'Clean Test Loss', self.test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'Clean Test Accuracy', self.test_accuracy.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'PGD Test Loss', self.PGD_test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'PGD Test Accuracy', self.PGD_test_accuracy.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'Test Global Robustness Loss',
                        self.global_robustness_test_loss.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'Test Global Robustness Violation Rate',
                        self.global_robustness_test_violation_rate.result(),
                        step=_epoch
                    )

            print(
                f'Epoch {_epoch + 1}, '
                f'Training Loss: {self.train_loss.result(): .3f}, '
                f'Training Accuracy: {self.train_accuracy.result() * 100: .2f}, '
                f'Training Global Robustness Loss: {self.global_robustness_train_loss.result(): .3f}, '
                f'Training Global Robustness Violation Rate: {self.global_robustness_train_violation_rate.result() * 100: .2f}, '
                f'Clean Test Loss: {self.test_loss.result(): .3f}, '
                f'Clean Test Accuracy: {self.test_accuracy.result() * 100: .2f}, '
                f'PGD Test Loss: {self.PGD_test_loss.result(): .3f}, '
                f'PGD Test Accuracy: {self.PGD_test_accuracy.result() * 100: .2f}, '
                f'Test Global Robustness Loss: {self.global_robustness_test_loss.result(): .3f}, '
                f'Test Global Robustness Violation Rate: {self.global_robustness_test_violation_rate.result() * 100: .2f}, '
                f'Training time: {time.time() - t: .2f}s',
                flush=True
            )

        file_name = "pgd_eps_{}_{}_steps_{}_{}_output_bound_{}_loss_coeff_{}_points_{}_{}_epoch_{}_batch_{}_lr_{}"\
            .format(self.pgd_train.eps, self.pgd_test.eps,
                    self.pgd_train.steps, self.pgd_test.steps,
                    self.global_robustness_output_bound,
                    self.global_robustness_coeff,
                    self.global_robustness_training_num_of_points,
                    self.global_robustness_test_num_of_points,
                    self.epoch, self.batch_size, self.lr)

        if self.target_class is not None:
            file_name += '_target_class_' + str(self.target_class)
        # Save the trained model
        self.save_model(file_name)

    def train_global_pgd_fine_tuning(self, only_final_evaluation=False):
        """
        Adversarial training for the global robustness property.
        """
        for _epoch in range(self.epoch):
            t = time.time()
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.global_robustness_train_loss.reset_states()
            self.global_robustness_test_loss.reset_states()
            self.global_robustness_train_violation_rate.reset_states()
            self.global_robustness_test_violation_rate.reset_states()

            self.AT_G_fine_tune_step()

            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    'Training Global Robustness Loss',
                    self.global_robustness_train_loss.result(),
                    step=_epoch
                )
                tf.summary.scalar(
                    'Training Global Robustness Violation Rate',
                    self.global_robustness_train_violation_rate.result(),
                    step=_epoch
                )

            if not only_final_evaluation or _epoch == self.epoch - 1:
                for x, y in self.test_ds:
                    # Test with PGD perturbed inputs.
                    perturbed_x = self.pgd_test.generate(x, log=False)
                    self.test_step(x, y, perturbed_x=perturbed_x)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar(
                        'Clean Test Loss', self.test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'Clean Test Accuracy', self.test_accuracy.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'PGD Test Loss', self.PGD_test_loss.result(), step=_epoch
                    )
                    tf.summary.scalar(
                        'PGD Test Accuracy', self.PGD_test_accuracy.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'Test Global Robustness Loss',
                        self.global_robustness_test_loss.result(),
                        step=_epoch
                    )
                    tf.summary.scalar(
                        'Test Global Robustness Violation Rate',
                        self.global_robustness_test_violation_rate.result(),
                        step=_epoch
                    )

            print(
                f'Epoch {_epoch + 1}, '
                f'Training Global Robustness Loss: {self.global_robustness_train_loss.result(): .3f}, '
                f'Training Global Robustness Violation Rate: {self.global_robustness_train_violation_rate.result() * 100: .2f}, '
                f'Clean Test Loss: {self.test_loss.result(): .3f}, '
                f'Clean Test Accuracy: {self.test_accuracy.result() * 100: .2f}, '
                f'PGD Test Loss: {self.PGD_test_loss.result(): .3f}, '
                f'PGD Test Accuracy: {self.PGD_test_accuracy.result() * 100: .2f}, '
                f'Test Global Robustness Loss: {self.global_robustness_test_loss.result(): .3f}, '
                f'Test Global Robustness Violation Rate: {self.global_robustness_test_violation_rate.result() * 100: .2f}, '
                f'Training time: {time.time() - t: .2f}s',
                flush=True
            )

        file_name = "fine_tuned_pgd_eps_{}_{}_steps_{}_{}_output_bound_{}_loss_coeff_{}_points_{}_{}_epoch_{}_lr_{}"\
            .format(self.pgd_train.eps, self.pgd_test.eps,
                    self.pgd_train.steps, self.pgd_test.steps,
                    self.global_robustness_output_bound,
                    self.global_robustness_coeff,
                    self.global_robustness_training_num_of_points,
                    self.global_robustness_test_num_of_points,
                    self.epoch, self.lr)
        if self.target_class is not None:
            file_name += '_target_class_' + str(self.target_class)
        # Save the trained model
        self.save_model(file_name)

    def save_model(self, file_name):
        model_json = model.to_json()
        netname = 'model_' + self.dataset + '_' + file_name
        with open(self.base_dir + netname + ".json", 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(self.base_dir + netname + ".h5")
        print("Saved model to disk")


def MPG_dataset():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    input_lb = np.array(train_features.describe().transpose()['min']).astype("float32")
    input_ub = np.array(train_features.describe().transpose()['max']).astype("float32")

    train_features = np.array(train_features).astype("float32")
    test_features = np.array(test_features).astype("float32")
    train_labels = np.array(train_labels).astype("float32")[..., tf.newaxis]
    test_labels = np.array(test_labels).astype("float32")[..., tf.newaxis]

    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_features)

    return train_features, train_labels, test_features, test_labels, normalizer, input_lb, input_ub

if __name__ == '__main__':
    # Parse input arguments
    args = parse_args()
    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dataset == 'mnist':
        # Prepare MNIST dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 3, 2, activation='relu'),
            tf.keras.layers.Conv2D(32, 3, 2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # Specify the whole input space as the input region.
        if args.target_class is None:
            input_lb = tf.zeros_like(x_train[0])
            input_ub = tf.ones_like(x_train[0])
        else:
            import sys
            from os.path import dirname, abspath
            sys.path.append(dirname(dirname(abspath(__file__))))
            from mnist_class_cube import mnist_class_cube
            input_lb, input_ub = mnist_class_cube(args.target_class)
            input_lb, input_ub = tf.cast(input_lb, tf.float32)/255.0, \
                tf.cast(input_ub, tf.float32)/255.0

    if args.dataset == 'cifar10':
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, 2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        input_lb = tf.zeros_like(x_train[0])
        input_ub = tf.ones_like(x_train[0])


    elif args.dataset == 'cryptojacker':
        # Prepare cryptojacker dataset
        x_train = np.loadtxt(
            './data/cryptojacker.train.csv', delimiter=',',
            usecols=list(range(1, 7 + 1))
        ).astype(np.float32)
        y_train = np.loadtxt(
            './data/cryptojacker.train.csv', delimiter=',', usecols=0
        ).astype(np.int)

        x_test = np.loadtxt(
            './data/cryptojacker.test.csv', delimiter=',',
            usecols=list(range(1, 7 + 1))
        ).astype(np.float32)
        y_test = np.loadtxt(
            './data/cryptojacker.test.csv', delimiter=',', usecols=0
        ).astype(np.int)

        # Max feature value.
        featmax = np.loadtxt(
            './data/cryptojacker_featstd.csv', delimiter=',',
            usecols=list(range(7))).astype(np.float32)
        # Normalize input to [0, 1]
        x_train, x_test = x_train / featmax, x_test / featmax

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(7,)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Specify the whole input space as the input region.
        input_lb = tf.zeros_like(x_train[0])
        input_ub = tf.ones_like(x_train[0])

    elif args.dataset == 'twitter_spam_account':
        # Prepare Twitter Spam Account dataset
        x_train = np.loadtxt(
            './data/social_honeypot.train.csv', delimiter=',',
            usecols=list(range(1, 15 + 1))
        ).astype(np.float32)
        y_train = np.loadtxt(
            './data/social_honeypot.train.csv', delimiter=',', usecols=0
        ).astype(np.int)

        x_test = np.loadtxt(
            './data/social_honeypot.test.csv', delimiter=',',
            usecols=list(range(1, 15 + 1))
        ).astype(np.float32)
        y_test = np.loadtxt(
            './data/social_honeypot.test.csv', delimiter=',', usecols=0
        ).astype(np.int)

        # Max feature value.
        featmax = np.loadtxt(
            './data/social_honeypot_featstd.csv', delimiter=',',
            usecols=list(range(15))).astype(np.float32)
        # Normalize input to [0, 1]
        x_train, x_test = x_train / featmax, x_test / featmax

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(15,)),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Specify the whole input space as the input region.
        input_lb = tf.zeros_like(x_train[0])
        input_ub = tf.ones_like(x_train[0])


    elif args.dataset == 'twitter_spam_url':
        # Prepare Twitter Spam Account dataset
        x_train = np.loadtxt(
            './data/unnormalized_twitter_spam.train.csv', delimiter=',',
            usecols=list(range(1, 25 + 1))
        ).astype(np.float32)
        y_train = np.loadtxt(
            './data/unnormalized_twitter_spam.train.csv', delimiter=',',
            usecols=0
        ).astype(np.int)

        x_test = np.loadtxt(
            './data/unnormalized_twitter_spam.test.csv', delimiter=',',
            usecols=list(range(1, 25 + 1))
        ).astype(np.float32)
        y_test = np.loadtxt(
            './data/unnormalized_twitter_spam.test.csv', delimiter=',',
            usecols=0
        ).astype(np.int)

        # Max feature value.
        featmax = np.loadtxt(
            './data/unnormalized_twitter_spam_featstd.csv', delimiter=',',
            usecols=list(range(25))).astype(np.float32)
        # Normalize input to [0, 1]
        x_train, x_test = x_train / featmax, x_test / featmax

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(25,)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Specify the whole input space as the input region.
        input_lb = tf.zeros_like(x_train[0])
        input_ub = tf.ones_like(x_train[0])

    elif args.dataset == 'german_credit_numeric':
        # Obtain features and labels of german_credit_numeric.
        # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
        features, labels = tfds.as_numpy(tfds.load(
            'german_credit_numeric', split='train',
            shuffle_files=True, batch_size=-1, as_supervised=True
        ))

        # Training and testing split ratio: 700/200
        x_train, x_test = features[:700].astype(np.float32), \
            features[700:].astype(np.float32)
        y_train, y_test = labels[:700], labels[700:]

        # Max feature value.
        featmax = np.loadtxt(
            './data/german_credit_numeric_featstd.csv', delimiter=',',
            usecols=list(range(24))).astype(np.float32)

        # Normalize the features.
        x_train, x_test = x_train / featmax, x_test / featmax

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(24,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Specify the whole input space as the input region.
        input_lb = None
        input_ub = None

    elif args.dataset == 'german_credit':
        # Obtain features and labels of german_credit.
        # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
        # Prepare Twitter Spam Account dataset
        features = np.loadtxt(
            './data/german_credit_redone.csv', delimiter=',',
            usecols=list(range(20))
        ).astype(np.float32)
        labels = np.loadtxt(
            './data/german_credit_redone.csv', delimiter=',',
            usecols=-1
        ).astype(np.int)

        # Training and testing split ratio: 700/200
        x_train, x_test = features[:700].astype(np.float32), \
            features[700:].astype(np.float32)
        y_train, y_test = labels[:700], labels[700:]

        # Min and Max feature value.
        featstd = np.loadtxt(
            './data/german_credit_featstd.csv', delimiter=',',
            usecols=list(range(20))).astype(np.float32)

        featmin_ = featstd[0, :]
        featmax_= featstd[1, :]
        featstd_ = featmax_ - featmin_

        # Normalize the features.
        x_train, x_test = (x_train - featmin_) / featstd_, \
            (x_test - featmin_) / featstd_

        # Initialize a keras model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(20,)),
            tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
            tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
            tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        ])

        # Specify the whole input space as the input region.
        MIN_AGE = 19
        MAX_AGE = 75
        AGE_INDEX = 12

        # Specify the whole input space as the input region.
        age_range_lb = args.age_range_lb 
        age_range_ub = args.age_range_ub

        input_lb = np.zeros(shape=20)
        input_lb[AGE_INDEX] = (age_range_lb - MIN_AGE) / (MAX_AGE - MIN_AGE)
        input_lb = tf.convert_to_tensor(input_lb, np.float32)

        input_ub = np.ones(shape=20)
        input_ub[AGE_INDEX] = (age_range_ub - MIN_AGE) / (MAX_AGE - MIN_AGE)
        input_ub = tf.convert_to_tensor(input_ub, np.float32)

    elif args.dataset == 'auto_mpg':
        # x_train, y_train, x_test, y_test, normalizer, input_lb, input_ub = MPG_dataset()
        with open('./data/auto_mpg_dataset.pickle', 'rb') as f:
            x_train, y_train, x_test, y_test, input_lb, input_ub = pickle.load(f)
        normalizer = preprocessing.Normalization()
        normalizer.adapt(x_train)

        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    else:
        raise NotImplementedError(
            'Dataset, {}, has not been implemented.'.format(args.dataset)
        )

    if args.preload_netname is not None:
        if args.training_mode == 'ST-AT-G':
            model = load_model(
                './model/' + args.dataset + '_clean' \
                + '/' + args.preload_netname
            )
        elif args.training_mode == 'AT-AT-G':
            model = load_model(
                './model/' + args.dataset + '_local_adv' \
                + '/' + args.preload_netname
            )
        else:
            raise NotImplementedError(
                'Training mode, {}, is not supported for fine tuning'.format(
                    args.training_mode
                ))

    if args.dataset == 'german_credit':
        single_idx = 12
    else:
        single_idx = None

    trainner = PGD_train(
        model, args.dataset, x_train, y_train, x_test, y_test,
        eps_train=args.eps_train, eps_test=args.eps_test,
        eps_step=args.eps_step, PGD_steps_train=args.PGD_steps_train,
        PGD_steps_test=args.PGD_steps_test, epoch=args.epoch,
        batch_size=args.batch_size, binary=(args.dataset not in ['mnist', 'german_credit', 'cifar10']),
        lr=args.learning_rate, training_mode=args.training_mode,
        input_lb=input_lb, input_ub=input_ub,
        global_robustness_training_num_of_points=args.global_robustness_training_num_of_points,
        global_robustness_test_num_of_points=args.global_robustness_test_num_of_points,
        global_robustness_coeff=args.global_robustness_coeff,
        global_robustness_output_bound=args.global_robustness_output_bound,
        regresssion=(args.dataset == 'auto_mpg'),
        fine_tune=(args.preload_netname is not None),
        target_class=args.target_class,
        single_idx=single_idx
    )
    t0 = time.time()
    # Train with the specified training mode
    if args.training_mode == 'clean':
        # Perform nomral training
        trainner.train(args.only_final_evaluation)
    elif args.training_mode == 'local_adv':
        # Perform PGD training
        trainner.train_pgd(args.only_final_evaluation)
    elif args.training_mode == 'global_adv':
        # Perform PGD training for the global robusntess
        trainner.train_global_pgd(args.only_final_evaluation)
    elif args.training_mode == 'ST-AT-G':
        # Perform AT-G fine tuning for ST model.
        trainner.train_global_pgd_fine_tuning(args.only_final_evaluation)
    elif args.training_mode == 'AT-AT-G':
        # Perform AT-G fine tuning for local AT model.
        trainner.train_global_pgd_fine_tuning(args.only_final_evaluation)

    else:
        raise NotImplementedError(
            'Training mode, {}, has not been implemented'.format(
                args.training_mode
            ))
    print("Training time: ", time.time() - t0)