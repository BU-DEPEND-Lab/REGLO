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
    netname = 'model_itne_mnist_' + fname
    pickle_filename = 'ITNE_model/model/' + netname + '.pickle'
    extracted_layers = None
    with open(pickle_filename, 'rb') as f:
        extracted_layers = pickle.load(f)
    return extracted_layers

def main(model_name, idx, batch_size=64):
    input_lb = imageio.imread("data/mnist_lb.png") / 255.0
    input_ub = imageio.imread("data/mnist_ub.png") / 255.0

    print("MNIST model", model_name, "results will be saved in", f'results/results0725_mnist_{idx}.pickle')

    eps = 2.0/255.0
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    Layers = load_extracted_model(model_name)

    test_bnb = BnB(model_name, batch_size=batch_size)
    test_bnb.setupNeuralNetwork(Layers, diff_lb, diff_ub)
    res = test_bnb.branchnBound(n_splits=20000, timeout=30_000)
    # print(res)
    with open(f'ITNE_model/results/results0725_mnist_{idx}.pickle', 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    # models = ['1c2d_reg_1e_2_0', '2c2d_reg_1e_2_0', '3c2d_reg_1e_2_0']
    models = ['1c2d', '2c2d', '3c2d']
    for i in range(1,2):
        main(models[i], i, batch_size=(3-i)*32)

"""
nohup python -m ITNE_model.eval_mnist > ITNE_model/results/log0725_0.log 2>&1 &
"""