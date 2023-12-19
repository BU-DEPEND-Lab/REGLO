import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

class Reglo_MPG(Reglo):
    def __init__(self, netname:str) -> None:
        tf_model = load_model(netname)
        with open('./adversarial_training/data/auto_mpg_dataset.pickle', 'rb') as f:
            x_train, y_train, x_test, y_test, input_lb, input_ub = pickle.load(f)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(500).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        super().__init__(tf_model, input_lb, input_ub, delta=5e-2, eps=1.85, area_search='milp', milp_max_regions=10, dataset=(train_ds, test_ds), model_name=netname.split('/')[-1])

    def eval_accuracy(self):
        """
        Evaluate the accuracy of self.model according to the testing data in self.dataset
        Result is appended to the end of self.results['accuracy']
        """
        _, test_ds = self.dataset
        test_accuracy = tf.keras.metrics.MeanAbsoluteError()
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

    def iterative_repair(self, target_eps):
        grad = self.milp_worst_gradient()
        self.milp_iteration = 2
        step = 0
        while(grad > target_eps/self.delta):
            self.eps = 0.95 * grad * self.delta
            self.grad_threshold = self.eps/self.delta
            self.repair_one_step(repair_steps=1000, repair_step_size=0.1)
            acc = self.eval_accuracy()
            print("Repair step", step, "Testing Accuracy", acc)
            grad = self.milp_worst_gradient()
            step += 1


if __name__ == '__main__':
    # netname = './adversarial_training/model/auto_mpg_clean/20220814-162352/model_auto_mpg_epoch_100_batch_32_lr_0.001'
    # netname = './adversarial_training/model/auto_mpg_global_adv/20220814-162634/model_auto_mpg_pgd_eps_0.05_0.05_steps_10_200_output_bound_1.5_loss_coeff_0.1_points_100_1000_epoch_100_batch_32_lr_0.001'
    netname = './adversarial_training/model/auto_mpg_ST-AT-G_fine_tuning/20220815-225055/model_auto_mpg_fine_tuned_pgd_eps_0.05_0.05_steps_10_200_output_bound_1.5_loss_coeff_0.1_points_100_1000_epoch_100_lr_0.0001'
    reglo_mpg = Reglo_MPG(netname)
    acc = reglo_mpg.eval_accuracy()
    print("Testing Accuracy", acc)
    reglo_mpg.eval_pgd(num_of_points=2000, out_bound=1.5)
    reglo_mpg.eval_global_robustness(n_splits=100_000, timeout=600, batch_size=2048)
    # t0 = time.time()
    # # reglo_mpg.repair_one_step(repair_steps=1000, repair_step_size=0.1)
    # reglo_mpg.iterative_repair(target_eps=1.5)
    # print("repair time:", time.time() - t0)
    # # acc = reglo_mpg.eval_accuracy()
    # # print("Testing Accuracy", acc)
    # # reglo_mpg.milp_worst_gradient()
    # reglo_mpg.eval_pgd(num_of_points=2000, out_bound=1.5)
    # reglo_mpg.eval_global_robustness(n_splits=100_000, timeout=1200, batch_size=4096)


