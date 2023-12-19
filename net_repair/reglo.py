import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

import sys
# setting path
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from GlobalRobust_beta import nn_extractor, RP_GlobR
from BnB import BnB

from mnist_sample import random_sample
from milp_init import global_gradient
from ITNE_CROWN.itnecrown_NeurNet import NeurNet
from repair_core import Repair

import numpy as np
import pickle
import time

class Reglo:
    def __init__(
        self, tf_model, input_lb, input_ub,
        delta=1e-2, eps = 1, fairness_index = -1,
        area_search='sample', milp_max_regions=10,
        sample_mode="dataset", dataset=None, model_name=''
    ) -> None:
        self.model = tf_model
        self.extracted_model = nn_extractor(self.model)
        self.grad_threshold = eps/delta
        self.area_search_mode = area_search
        self.milp_iteration = milp_max_regions
        self.sample_mode = sample_mode
        self.dataset = dataset
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.delta = delta
        self.region_delta = delta * 2
        self.eps = eps
        self.fairness_index = fairness_index
        self.model_name = model_name
        self.results = {}
        self.reset_result_table()
        self.repair_regions = []    # centers of repair regions

    def reset_result_table(self):
        self.results['accuracy'] = []
        self.results['pgd accuracy'] = []
        self.results['global robustness'] = []

    def save_results(self):
        save_path = f"./results/reglo_res_{self.model_name}.pickle"
        with open(save_path, 'wb') as f:
            pickle.dump(self.results, f)

    def violated_regions(self):
        if self.area_search_mode == 'sample':
            return self._sample_violated_regions()
        else:
            return self._milp_violated_regions()

    def _sample_violated_regions(self, is_random=True):
        train_ds, test_ds = self.dataset
        hist, _ = random_sample(self.model, train_ds, test_ds, self.input_lb, self.input_ub, data_from=self.sample_mode, random_multiplier=1, threshold=self.grad_threshold, fairness_index=self.fairness_index)
        violated_areas_num = len(hist.large_grad_samples)
        if is_random:
            import random
            my_choise = random.sample(range(violated_areas_num), self.milp_iteration)
        else:
            my_choise = range(min(violated_areas_num, self.milp_iteration))
        print(f'Total number of selected samples: {len(hist.large_grad_samples)}, pick {self.milp_iteration} of them.')
        return [hist.large_grad_samples[i].numpy() for i in my_choise]

    def _milp_violated_regions(self):
        _, res = global_gradient(
            self.extracted_model, self.input_lb, self.input_ub,
            output_dim=self.extracted_model[-1]['output_shape'],
            max_points=self.milp_iteration,
            threshold=self.grad_threshold,
            fairness_index=self.fairness_index
        )
        return res

    def milp_worst_gradient(self):
        grads, samples = global_gradient(
            self.extracted_model, self.input_lb, self.input_ub,
            output_dim=self.extracted_model[-1]['output_shape'],
            max_points=1,
            threshold=self.grad_threshold
        )
        return grads[0]

    def sample_worst_gradient(self):
        train_ds, test_ds = self.dataset
        hist, _ = random_sample(self.model, train_ds, test_ds, self.input_lb, self.input_ub, data_from=self.sample_mode, random_multiplier=1, threshold=self.grad_threshold)
        return hist.max


    def last_hidden_layer_bounds(self, violated_region_sample):
        sample = np.array(violated_region_sample)
        if self.fairness_index < 0:
            region_delta = self.region_delta
        else:
            region_delta = np.zeros_like(sample)
            region_delta[self.fairness_index] = self.region_delta
        input_lb = np.clip(sample - region_delta, self.input_lb, self.input_ub)
        input_ub = np.clip(sample + region_delta, self.input_lb, self.input_ub)
        if self.fairness_index < 0:
            diff_lb = -self.delta * np.ones_like(input_lb)
            diff_ub = self.delta * np.ones_like(input_lb)
        else:
            diff_lb = np.zeros_like(input_lb)
            diff_ub = np.zeros_like(input_lb)
            diff_lb[self.fairness_index] = -self.delta
            diff_ub[self.fairness_index] = self.delta
        itne_nn = NeurNet("reglo_model", alpha=True)
        itne_nn.setupNeuralNetwork(self.extracted_model[:-1], input_lb, input_ub, diff_lb, diff_ub)
        _, _, _, _, dA_lb, dA_ub, bias_lb, bias_ub = itne_nn.bound_backward()
        return dA_lb.numpy(), dA_ub.numpy(), bias_lb.numpy(), bias_ub.numpy()

    def repair_one_step(self, repair_steps=100, repair_step_size=0.3, t=1):
        violated_samples = self.violated_regions()
        areas_num = len(violated_samples)
        input_lb = np.array(self.input_lb).flatten()
        input_ub = np.array(self.input_ub).flatten()
        in_dim = len(input_ub)
        if self.fairness_index < 0:
            dx_set = [np.block([[np.eye(in_dim)], [-np.eye(in_dim)]]), self.delta * np.ones(2*in_dim)]
        else:
            offset = np.zeros(2*in_dim)
            offset[self.fairness_index] = self.delta
            offset[in_dim + self.fairness_index] = self.delta
            dx_set = [np.block([[np.eye(in_dim)], [-np.eye(in_dim)]]), offset]
        out_weights = self.extracted_model[-1]['kernel'].T
        self.repair_regions = []
        repair_engine = Repair(dx_set, self.eps, out_weights)
        for sample in violated_samples:
            dA_lb, dA_ub, bias_lb, bias_ub = self.last_hidden_layer_bounds(sample)
            repair_engine.add_constraints([dA_lb, dA_ub, bias_lb, bias_ub])
            self.repair_regions.append(sample)
        # print(dA_lb.shape, bias_lb.shape, out_weights.shape)
        start_time = time.time()
        delta_theta = repair_engine.repair(repair_steps, step_size=repair_step_size, t=t*areas_num).T
        print(f'Repair time: {time.time()-start_time:.2f}')
        self.extracted_model[-1]['kernel'] += delta_theta
        self.model.layers[-1].kernel.assign(self.extracted_model[-1]['kernel'])


    def eval_accuracy(self):
        """
        Evaluate the accuracy of self.model according to the testing data in self.dataset
        Result is appended to the end of self.results['accuracy']
        """
        # NOTE: This only suitable for classification problems (e.g., MNIST)
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        _, test_ds = self.dataset
        @tf.function
        def test_step(images, labels):
            predictions = self.model(images, training=False)
            test_accuracy(labels, predictions)

        test_accuracy.reset_states()
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        acc = test_accuracy.result()
        self.results['accuracy'].append(acc)
        return acc

    def eval_pgd(self, num_of_points, out_bound=None, single_idx=None):
        """
        Evaluate the accuracy of self.model under PGD attack according to the testing data in self.dataset
        Result is appended to the end of self.results['pgd accuracy']
        """
        if out_bound is None:
            out_bound = self.eps
        from net_repair.adversarial_training.PGD import ProjectedGradientDescentForRegion
        # Obtain PGD perturbed inputs within the input region
        PGD_on_input_region = ProjectedGradientDescentForRegion(
            model=self.model, input_lb=self.input_lb, input_ub=self.input_ub,
            eps=self.delta, eps_step=0.1, steps=200
        )

        perturbed_inputs, clean_inputs = PGD_on_input_region.random_sample(
            num_of_points=num_of_points, single_idx=single_idx
        )
        diff = self.model(perturbed_inputs) - self.model(clean_inputs)

        # Compute the percentage of points that violate the global robustness
        # specification.
        violation_rate = tf.math.reduce_sum(tf.cast(tf.math.greater(
            tf.norm(diff, ord=np.inf, axis=-1), out_bound), tf.float32)) \
            / num_of_points

        print("============Global robustness violation rate is {}% among {} "
              "samples.============".format(violation_rate * 100, num_of_points))

        global_robustness_bound = tf.reduce_max(tf.norm(diff, ord=np.inf, axis=-1))

        print(f"============Global robustness bound is {global_robustness_bound:.4f}.============")

    def eval_global_robustness(self, use_latest_ITNECROWN=True, timeout=500, n_splits=20000, batch_size=128, max_refine_per_layer=20):
        """
        Evaluate the global robustness in the input domain [self.input_lb, self.input_ub] according to self.extracted_model
        Result is appended to the end of self.results['global robustness']

        When use_latest_ITNECROWN:
            timeout: ITNECROWN can stop at anytime. Set a timeout for the algorithm to terminate.
            n_splits: ITNECROWN will terminate after searching n_splits splits (also a termination condition).
            batch_size: number of splits that will be handle simutaniously. "Larger batch_size => faster". Mainly depends on the GPU capacity.

        When not use_latest_ITNECROWN (use DATE'22 certification algorithm):
            timeout: a upper time limit to evaluate the bounds for ONE LAYER (NOT ENTIRE NETWORK)
            max_refine_per_layer: max number of neurons to refine. Controls the number of integer variables in MILP as well as the level of over-approximation.
        """
        input_lb = self.input_lb
        input_ub = self.input_ub
        if isinstance(input_lb, tf.Tensor):
            input_lb = input_lb.numpy()
        if isinstance(input_ub, tf.Tensor):
            input_ub = input_ub.numpy()
        if self.fairness_index < 0:
            diff_lb = -self.delta * np.ones_like(input_lb)
            diff_ub = self.delta * np.ones_like(input_lb)
        else:
            diff_lb = np.zeros_like(input_lb)
            diff_ub = np.zeros_like(input_lb)
            diff_lb[self.fairness_index] = -self.delta
            diff_ub[self.fairness_index] = self.delta

        if use_latest_ITNECROWN:
            test_bnb = BnB(self.model_name, batch_size=batch_size)
            test_bnb.setupNeuralNetwork(self.extracted_model, input_lb, input_ub, diff_lb, diff_ub)
            res = test_bnb.branchnBound(n_splits=n_splits, timeout=timeout)
            print("Output variation lower bound:", res['lb'][-1])
            print("Output variation upper bound:", res['ub'][-1])
        else:
            rp_globR = RP_GlobR(self.extracted_model, dbg = False, timeout = timeout, lp_relax = True, nThreads = 16)
            win_sz = 3
            max_x = max_diff = max_refine_per_layer
            rp_globR.set_refine_config(win_sz, 0.8, max_diff, max_x=max_x)
            output_range = rp_globR.range_propergation_3(input_lb, input_ub, diff_lb, diff_ub,
                                                        window_size = win_sz)
            print("output variation bounds:", output_range)


