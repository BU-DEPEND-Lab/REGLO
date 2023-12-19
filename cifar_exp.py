import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from BnB import BnB
from NeurNet import load_extracted_model
import imageio
import numpy as np
import pickle

def main():
    model_name = '4c3d_noBN_reg_1e_3_0'
    input_lb = imageio.imread("data/cifar_lb.png") / 255.0
    input_ub = imageio.imread("data/cifar_ub.png") / 255.0

    eps = 1e-3
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    Layers = load_extracted_model(model_name)

    # for layer in Layers:    # Used for checkng model shape and size
    #     if 'output_shape' in layer:
    #         print(layer['type'], layer['output_shape'])
    # return

    test_bnb = BnB(model_name, batch_size=12)
    test_bnb.setupNeuralNetwork(Layers, diff_lb, diff_ub)
    res = test_bnb.branchnBound(n_splits=20000, timeout=100_000)
    print(res)
    with open('results0317_0.pickle', 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    main()
