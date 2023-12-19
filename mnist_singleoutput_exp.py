import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from BnB import BnB
from NeurNet import load_extracted_model, load_model
import imageio
import numpy as np
import pickle

def main():
    eps = 1e-3
    input_lb = np.zeros((28,28,1))
    input_ub = np.ones((28,28,1))
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    # Layers = load_extracted_model(model_name)
    Layers, model_name = load_model()

    test_bnb = BnB(model_name, batch_size=64)
    test_bnb.setupNeuralNetwork(Layers, diff_lb, diff_ub)
    res = test_bnb.branchnBound(n_splits=20000)
    with open('results0316_1.pickle', 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    main()