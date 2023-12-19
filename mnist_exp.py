import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from BnB import BnB
# from NeurNet import load_extracted_model
import imageio
import numpy as np
import pickle

def load_extracted_model(fname):
    netname = 'model_mnist_' + fname
    pickle_filename = 'data/model/' + netname + '.pickle'
    extracted_layers = None
    with open(pickle_filename, 'rb') as f:
        extracted_layers = pickle.load(f)
    # for layer in extracted_layers:    # Used for checkng model shape and size
    #     if 'output_shape' in layer:
    #         print(layer['output_shape'])
    return extracted_layers

def main(model_name, idx, batch_size=64):
    input_lb = imageio.imread("data/mnist_lb.png") / 255.0
    input_ub = imageio.imread("data/mnist_ub.png") / 255.0

    print("MNIST model", model_name, "results will be saved in", f'results0429_mnist_{idx}.pickle')

    eps = 2.0/255.0
    np.random.seed(12345)
    sample = np.random.rand(28,28,1)
    input_lb = np.clip(sample - 5 * eps, 0, 1)
    input_ub = np.clip(sample + 5 * eps, 0, 1)
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    Layers = load_extracted_model(model_name)
    # return    # Used for checkng model shape and size

    test_bnb = BnB(model_name, batch_size=batch_size)
    test_bnb.setupNeuralNetwork(Layers, input_lb, input_ub, diff_lb, diff_ub)
    res = test_bnb.branchnBound(n_splits=20000, timeout=10_000)
    print(res)
    # with open(f'results0429_mnist_{idx}.pickle', 'wb') as f:
    #     pickle.dump(res, f)

if __name__ == '__main__':
    models = ['1c2d_reg_1e_2_0', '2c2d_reg_1e_2_0', '3c2d_reg_1e_2_0']
    for i in range(1,2):
        main(models[i], i, batch_size=(3-i)*32)

"""
nohup python mnist_exp.py > log0429_2.log 2>&1 &
"""