import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import model_from_json
from ITNE_CROWN.itnecrown_NeurNet import NeurNet

def load_model(fname='2c2d_reg_1e_2_0'):
    netname = 'model_mnist_'+fname
    json_filename = '../data/model/' + netname + '.json'
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights('../data/model/' + netname + '.h5')
    return dnn_model

def build_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # y_train, y_test = y_train % 2, y_test % 2

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        # (x_train, y_train)).batch(128)
        (x_train, y_train)).shuffle(10000).batch(256)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)
    return train_ds, test_ds

class histogram:
    def __init__(self, offset, width, n_bins, threshold=None) -> None:
        self.offset = offset
        self.width = width
        self.n_bins = n_bins
        self.n = 0
        self.hist = np.zeros(n_bins)
        self.below = 0
        self.above = 0
        self.max_val = -np.inf
        self.threshold = threshold
        self.large_grad_samples = []

    def update(self, vars: np.ndarray, samples):
        vars = vars.flatten()
        idxs = np.floor((vars - self.offset) / self.width).astype(int)
        # for i, v in zip(idxs, vars):
        for j in range(len(vars)):
            i = idxs[j]
            v = vars[j]
            if i < 0:
                self.below += 1
            elif i >= self.n_bins:
                self.above += 1
            else:
                self.hist[i] += 1
            self.n += 1
            self.max_val = max(self.max_val, v)
            if self.threshold is not None and v > self.threshold:
                self.large_grad_samples.append(samples[j])

    @property
    def max(self):
        return self.max_val

    @property
    def below_percentage(self):
        return self.below/self.n

    @property
    def above_percentage(self):
        return self.above/self.n

    def percentage_over(self, bound):
        return self.over(bound) / self.n
    
    def over(self, bound):
        res = 0
        for i in range(int((bound - self.offset)//self.width), self.n_bins):
            res += self.hist[i]
        res += self.above
        return res

    def plot(self, label):
        xs = self.offset + self.width/2 + self.width * np.arange(self.n_bins)
        probs = self.hist / (self.n * self.width)
        plt.plot(xs, probs, label=label)

    def globRobust_violate_rate(self, itne_nn: NeurNet):
        count = 0
        eps = 2/255
        diff_lb = diff_ub = None
        max_gr_grad = 0
        for sample in self.large_grad_samples:
            sample = sample.numpy()
            
            input_lb = np.clip(sample - 2*eps, 0.0, 1.0)
            input_ub = np.clip(sample + 2*eps, 0.0, 1.0)
            if diff_lb is None:
                diff_lb = -eps * np.ones_like(input_lb)
                diff_ub = eps * np.ones_like(input_lb)

            itne_nn.setupBounds(input_lb, input_ub, diff_lb, diff_ub)
            _, _, dlb, dub, _, _, _, _ = itne_nn.bound_backward()
            dlb, dub = dlb.numpy(), dub.numpy()
            gr_grads = np.abs(np.concatenate((dlb, dub))) / eps
            max_gr_grad = max(max_gr_grad, max(gr_grads))
            if (gr_grads > self.threshold).any():
                count += 1
            print('>', end='', flush=True)
        print('max output variation / delta =', max_gr_grad)
        return count / len(self.large_grad_samples)


def random_sample(model, train_ds, test_ds, input_lb, input_ub, 
                  data_from='dataset', 
                  grad_type='gradient', 
                  random_multiplier=100, 
                  threshold=None,
                  fairness_index=-1 # feature index to evaluate fairness (for multi-dim feature map, this is the index after flatten.)
                  ):

    print('mnist sample:', data_from, grad_type)

    input_diff = input_ub - input_lb

    def get_gradient(images):
        if fairness_index < 0:
            return get_jacobian(images)
        else:
            return get_fairness_jacobian(images)

    @tf.function
    def get_jacobian(images):
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = model(images)

        # Get the gradients of the loss w.r.t to the input image.
        jacobian = tape.batch_jacobian(predictions, images)
        shape = [i for i in jacobian.shape[:2]] + [-1]
        jacobian = tf.reshape(jacobian, shape)
        return tf.norm(jacobian, ord=np.inf, axis=[-2,-1])

    @tf.function
    def get_fairness_jacobian(images):
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = model(images)

        # Get the gradients of the loss w.r.t to the input image.
        jacobian = tape.batch_jacobian(predictions, images)
        shape = [i for i in jacobian.shape[:2]] + [-1]
        jacobian = tf.reshape(jacobian, shape)
        return tf.norm(jacobian[:, :, fairness_index], ord=np.inf, axis=-1)

    @tf.function
    def free_pgd(images, eps):
        delta = (tf.random.uniform(images.shape) - 0.5) * 2 * eps

        rate = 1e-1
        steps = 200
        for i in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(images)
                tape.watch(delta)
                diff = model(images + delta) - model(images)
                loss = tf.reduce_mean(tf.abs(diff))

            # Get the gradients of the loss w.r.t to the input image.
            grad, delta_grad = tape.gradient(loss, [images, delta])
            signed_grad = tf.sign(grad)
            images = images + rate * eps * signed_grad * 0.2
            images = tf.clip_by_value(images, 0, 1)
            signed_grad = tf.sign(delta_grad)
            delta = delta + rate * eps * signed_grad
            delta = tf.clip_by_value(delta, -eps, eps)
        diff = model(images + delta) - model(images)
        return tf.norm(diff, ord=np.inf, axis=-1) / eps

    @tf.function
    def pgd(images, eps):
        delta = (tf.random.uniform(images.shape) - 0.5) * 2 * eps

        rate = 1e-1
        steps = 50
        for i in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(delta)
                diff = model(images + delta) - model(images)
                loss = tf.reduce_mean(tf.abs(diff))

            # Get the gradients of the loss w.r.t to the input image.
            delta_grad = tape.gradient(loss, delta)
            signed_grad = tf.sign(delta_grad)
            delta = delta + rate * eps * signed_grad
            delta = tf.clip_by_value(delta, -eps, eps)
        diff = model(images + delta) - model(images)
        return tf.norm(diff, ord=np.inf, axis=-1) / eps

    hist = histogram(50, 0.1, 200, threshold)
    max_grads = []

    for train_images, _ in train_ds:
        if data_from == 'dataset':
            if grad_type == 'gradient':
                grad = get_gradient(train_images)
            elif grad_type == 'pgd':
                grad = pgd(train_images, 2/255)
            else:   # free PGD
                grad = free_pgd(train_images, 2/255)
            hist.update(grad.numpy(), train_images)
        else:   # random
            for _ in range(random_multiplier):
                random_images = input_diff * tf.random.uniform(train_images.shape) + input_lb
                if grad_type == 'gradient':
                    grad = get_gradient(random_images)
                elif grad_type == 'pgd':
                    grad = pgd(random_images, 2/255)
                else:   # free PGD
                    grad = free_pgd(random_images, 2/255)
                hist.update(grad.numpy(), random_images)

        print('.', end='', flush=True)
        max_grads.append(hist.max)

    for test_images, _ in test_ds:
        if data_from == 'dataset':
            if grad_type == 'gradient':
                grad = get_gradient(test_images)
            elif grad_type == 'pgd':
                grad = pgd(test_images, 2/255)
            else:   # free PGD
                grad = free_pgd(test_images, 2/255)
            hist.update(grad.numpy(), test_images)
        else:   # random
            for _ in range(random_multiplier):
                random_images = input_diff * tf.random.uniform(test_images.shape) + input_lb
                if grad_type == 'gradient':
                    grad = get_gradient(random_images)
                elif grad_type == 'pgd':
                    grad = pgd(random_images, 2/255)
                else:   # free PGD
                    grad = free_pgd(random_images, 2/255)
                hist.update(grad.numpy(), random_images)

        print('.', end='', flush=True)
        max_grads.append(hist.max)

    print()
    return hist, max_grads

def comparison(model, random_multiplier=100):
    train_ds, test_ds = build_dataset()
    input_lb = imageio.imread("../data/mnist_lb.png") / 255.0
    input_ub = imageio.imread("../data/mnist_ub.png") / 255.0
    input_lb = input_lb[..., tf.newaxis].astype("float32")
    input_ub = input_ub[..., tf.newaxis].astype("float32")
    hist_ds, max_ds_grads = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='gradient')
    hist_ds_pgd, max_ds_pgd_grads = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='pgd')
    hist_ds_fpgd, max_ds_fpgd_grads = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='fpgd')
    hist_rd, max_rd_grads = random_sample(
        model, train_ds, test_ds, input_lb, input_ub, data_from='random', grad_type='gradient', random_multiplier=random_multiplier)
    hist_rd_pgd, max_rd_pgd_grads = random_sample(
        model, train_ds, test_ds, input_lb, input_ub, data_from='random', grad_type='pgd', random_multiplier=random_multiplier)
    hist_rd_fpgd, max_rd_fpgd_grads = random_sample(
        model, train_ds, test_ds, input_lb, input_ub, data_from='random', grad_type='fpgd', random_multiplier=random_multiplier)
    print(hist_ds.max, hist_ds_pgd.max, hist_ds_fpgd.max, hist_rd.max, hist_rd_pgd.max, hist_rd_fpgd.max)
    print("Over 65:", hist_ds.over(65), hist_ds_pgd.over(65), hist_ds_fpgd.over(65), hist_rd.over(65), hist_rd_pgd.over(65), hist_rd_fpgd.over(65))
    print("Over 67:", hist_ds.over(67), hist_ds_pgd.over(67), hist_ds_fpgd.over(67), hist_rd.over(67), hist_rd_pgd.over(67), hist_rd_fpgd.over(67))
    print("Over 69:", hist_ds.over(69), hist_ds_pgd.over(69), hist_ds_fpgd.over(69), hist_rd.over(69), hist_rd_pgd.over(69), hist_rd_fpgd.over(69))

    plt.figure()
    plt.plot(max_ds_grads, label='dataset')
    plt.plot(max_rd_grads, label=f'x{random_multiplier} random')
    plt.plot(max_ds_pgd_grads, label='dataset pgd')
    plt.plot(max_ds_fpgd_grads, label='dataset fpgd')
    plt.plot(max_rd_pgd_grads, label=f'x{random_multiplier} random pgd')
    plt.plot(max_rd_fpgd_grads, label=f'x{random_multiplier} random fpgd')
    plt.legend()
    plt.savefig(f'mnist_sample_x{random_multiplier}.png')

    plt.figure()
    hist_ds.plot('dataset')
    hist_rd.plot(f'x{random_multiplier} random')
    hist_ds_pgd.plot('dataset pgd')
    hist_ds_fpgd.plot('dataset fpgd')
    hist_rd_pgd.plot(f'x{random_multiplier} random pgd')
    hist_rd_fpgd.plot(f'x{random_multiplier} random fpgd')
    plt.legend()
    plt.savefig(f'mnist_sample_x{random_multiplier}_probs.png')

def globRobust_violation_rate(model, desire_grad=65, random_multiplier=100):
    train_ds, test_ds = build_dataset()
    input_lb = imageio.imread("../data/mnist_lb.png") / 255.0
    input_ub = imageio.imread("../data/mnist_ub.png") / 255.0
    input_lb = input_lb[..., tf.newaxis].astype("float32")
    input_ub = input_ub[..., tf.newaxis].astype("float32")
    test_NN = NeurNet(model.name, alpha=True)
    test_NN.setupNNfromTFmodel(model)
    hist_ds, _ = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='gradient', threshold=desire_grad)
    print(hist_ds.globRobust_violate_rate(test_NN))
    # hist_rd, _ = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='random', grad_type='gradient', threshold=desire_grad)

def test_fairness_index(model):
    train_ds, test_ds = build_dataset()
    input_lb = imageio.imread("../data/mnist_lb.png") / 255.0
    input_ub = imageio.imread("../data/mnist_ub.png") / 255.0
    input_lb = input_lb[..., tf.newaxis].astype("float32")
    input_ub = input_ub[..., tf.newaxis].astype("float32")
    test_NN = NeurNet(model.name, alpha=True)
    test_NN.setupNNfromTFmodel(model)
    hist_ds, _ = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='gradient')
    print("max jacobian norm:", hist_ds.max)
    hist_ds, _ = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='gradient', fairness_index=12)
    print("max gradient norm for index 12:", hist_ds.max)
    hist_ds, _ = random_sample(model, train_ds, test_ds, input_lb, input_ub, data_from='dataset', grad_type='gradient', fairness_index=132)
    print("max gradient norm for index 132:", hist_ds.max)

if __name__ == '__main__':
    model = load_model()
    # comparison(model, 20)
    # globRobust_violation_rate(model, 69)
    test_fairness_index(model)