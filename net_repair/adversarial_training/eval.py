import os
import sys
import argparse
import imageio
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from PGD import ProjectedGradientDescentForRegion
# setting path
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='PGD evaluation for the global robustness.'
    )
    parser.add_argument('--dataset', type=str, help='Dataset name.',
                        required=True)
    parser.add_argument(
        '--input_bound', type=float, default=0.3,
        help='The bound for input of the global robustness property.',
        required=False
    )
    parser.add_argument(
        '--output_bound', type=float, default=0.3,
        help='The bound for output of the global robustness property.',
        required=False
    )
    parser.add_argument(
        '--num_of_points', type=int, default=100,
        help='The number of randomly sampled points in the input region for '
        'testing.',
        required=False
    )
    parser.add_argument(
        '--model_name', type=str,
        help='The name of the model for testing.', required=True
    )
    return parser.parse_args()


def load_model(fname='2c2d_reg_1e_2_0'):
    netname = 'model_mnist_'+fname
    json_filename = './model/' + netname + '.json'
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights('./model/' + netname + '.h5')
    return dnn_model


def PGD_eval(input_lb, input_ub,
             input_bound=0.3, output_bound=0.5,
             num_of_points=100, fname='2c2d_reg_1e_2_0'):
    """
    Evaluate with mnist.
    :param input_lb: The lower bound of the input region.
    :param input_ub: The upper bound of the input region.
    :param input_bound: The input bound from the global robustness
                        specification.
    :param output_bound: The output bound from the global robustness
                         specification.
    :param fname: The file name for the model.
    """
    # Load a model
    model = load_model(fname)

    # Obtain PGD perturbed inputs within the input region
    PGD_on_input_region = ProjectedGradientDescentForRegion(
        model=model, input_lb=input_lb, input_ub=input_ub,
        eps=input_bound, eps_step=0.1, steps=200
    )

    perturbed_inputs, clean_inputs = PGD_on_input_region.random_sample(
        num_of_points=num_of_points
    )
    diff = model(perturbed_inputs) - model(clean_inputs)

    # Compute the percentage of points that violate the global robustness
    # specification.
    violation_rate = tf.math.reduce_sum(tf.cast(tf.math.greater(
        tf.norm(diff, ord=np.inf, axis=-1), output_bound), tf.float32)) \
        / num_of_points

    print("============Global robustness violation rate is {}% among {} "
          "samples.============".format(violation_rate * 100, num_of_points))


if __name__ == '__main__':
    # Parse input arguments.
    args = parse_args()

    if args.dataset == 'mnist':
        # Specify input lower and upper bounds
        input_lb = imageio.imread("../../data/mnist_lb.png") / 255.0
        input_ub = imageio.imread("../../data/mnist_ub.png") / 255.0
        input_lb = input_lb[..., tf.newaxis].astype("float32")
        input_ub = input_ub[..., tf.newaxis].astype("float32")

        PGD_eval(
            input_lb, input_ub, args.input_bound, args.output_bound,
            args.num_of_points, args.model_name
        )
    else:
        raise NotImplementedError(
            'Dataset, {}, has not been implemented.'.format(args.dataset)
        )
