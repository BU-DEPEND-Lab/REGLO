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
    netname = 'model_itne_cifar_' + fname
    pickle_filename = 'ITNE_model/model/' + netname + '.pickle'
    extracted_layers = None
    with open(pickle_filename, 'rb') as f:
        extracted_layers = pickle.load(f)
    return extracted_layers

def main(model_name, idx, batch_size=64):
    input_lb = imageio.imread("data/cifar_lb.png") / 255.0
    input_ub = imageio.imread("data/cifar_ub.png") / 255.0

    print("CIFAR10 model", model_name, "results will be saved in", f'results/results0729_cifar_{idx}.pickle')

    eps = 1e-3
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    Layers = load_extracted_model(model_name)

    test_bnb = BnB(model_name, batch_size=batch_size)
    test_bnb.setupNeuralNetwork(Layers, diff_lb, diff_ub)
    res = test_bnb.branchnBound(n_splits=20000, timeout=100_000)
    # print(res)
    with open(f'ITNE_model/results/results0729_cifar_{idx}.pickle', 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    main('4c3d_0729', 0, batch_size=12)

"""
nohup python -m ITNE_model.eval_cifar > ITNE_model/results/log0727_0.log 2>&1 &
    - model name '4c3d'
    - After 100100.70114517212 s, final output variation bounds becomes 
        lbs = [-65.59998, -53.513386, -73.581436, -46.786064, -67.54241, -57.750965, -55.43357, -60.153236, -52.2144, -54.24802], 
        ubs = [65.47704, 53.483543, 75.817856, 47.278793, 67.52265, 58.03795, 55.741096, 69.96218, 52.193363, 54.17933].
nohup python -m ITNE_model.eval_cifar > ITNE_model/results/log0729_0.log 2>&1 &
    - model name '4c3d_0729'
    - After 100123.54092383385 s, final output variation bounds becomes 
        lbs = [-318.0749, -257.35468, -412.41187, -257.25925, -325.96036, -296.8431, -283.503, -337.45337, -251.16444, -220.30807], 
        ubs = [318.05438, 257.32648, 418.37735, 257.14246, 328.33694, 296.41043, 283.0289, 335.08923, 251.44414, 220.30348].
"""