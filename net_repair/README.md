# REGLO: Provable Neural Network Repair for Global Robustness Properties

## Step 1: Finding linear regions with largest gradient.

### MILP based method for small networks
The code can be found in [milp_init.py](milp_init.py), which trains a 20x20 relu network for the Auto MPG dataset. And looks for the first k activation patterns with largest gradient.

### Sampling method for large (MNIST) networks
The code can be found in [mnist_sample.py](mnist_sample.py). Multiple different sampling strategy is used:
- Dataset
    - evaluate the gradient for each sample in the dataset (including training and testing sets).
- Dataset PGD
    - for each sample $x$ in the dataset, using PGD to find a perturbed sample $x'$ where $\|x-x'\|\leq \varepsilon$, which maximize $\|f(x) - f(x')\|$.
- Dataset FPGD
    - FPGD is for free PGD, which has following steps
        - initialize $x$ to one of samples in the dataset. 
        - Use PGD to find both the un-perturbed sample $x$ and the perturbed sample $x'$ where $\|x-x'\|\leq \varepsilon$, which maximize $\|f(x) - f(x')\|$.
- $\times n$ Random
    - Evaulate the gradient of randomly sampled points. The number of randomly generated data is $n$ times of the dataset.
- $\times n$ Random PGD
    - similar to Dataset PGD but consider each randomly generated sample instead.
- $\times n$ Random FPGD
    - similar to Dataset FPGD but consider each randomly generated sample instead.

## Step 2: Evaluation of last hidden layer neuron difference (linear) bounds

The code can be found in [ITNE_CROWN/NeurNet.py](ITNE_CROWN/NeurNet.py). An example of evaluating the linear bound matrix and bias can be found in the `main()` function in [ITNE_CROWN/NeurNet.py](ITNE_CROWN/NeurNet.py). The code was designed for deriving the bounds of output layer. So if you need the bounds of the last hidden layer, just discard the output layer after nn extraction (details can be found in the `main()` function in [ITNE_CROWN/NeurNet.py](ITNE_CROWN/NeurNet.py)).

Currently the interface between step 1 and 2 are not implemented yet. For step 2, both the input $x$ and perturbation $\Delta x$ needs to have L-$\infty$ bound. For other types of bounds for input $x$, further discussion is needed.

The output variation bound is derived by ITNE-CROWN with no optimization (i.e. PGD) needed. The neural network output range is derived by $\alpha$-CROWN (needs PGD). If the speed is not acceptable, we can further simplify it to CROWN (not implemented yet).

# Experiment results:

## German Creadit
```
python reglo_german.py
```

## MNIST
```
python reglo_mnist.py
```

## Auto MPG
```
python reglo_mpg.py
```
