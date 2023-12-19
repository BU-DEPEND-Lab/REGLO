# Global robustness certification leveraging ITNE encoding and beta-CROWN
This method can provide an over-approximation (upper bound) of the neural network global robustness. The robustness is defined on the entire, infinite input domain. The branch-and-bound method substitudes the linear relaxation of ReLU distance into a still-over-approximated but-tighter over-approximation. Therefore, this method is incomplete (but sound).

## Related packages:
- Python 3.8
- Tensorflow 2.2
    - Most used APIs are standard TF2 API. One thing should be noticed is that in TF2.2, Keras Normalization layer (used in [ITNE_model/Normalization_layer_ITNE.py](ITNE_model/Normalization_layer_ITNE.py)) is located at `tf.keras.layers.experimental.preprocessing.Normalization`. However, in the latest TF2.9, it is located at `tf.keras.layers.Normalization`.
- Gurobi 9.1
    - The latest Gurobi should work. Please let me know if there are any API changes.

## Structure of this repo
- Root Directory
    - [GlobalRobust_beta.py](GlobalRobust_beta.py)
        - Original MILP-based global robustness certification algorithm (DATE'22).
        - Tensorflow models can be extracted by the `nn_extractor()` and used in other global robustness related algorithms.
    - [BnB.py](BnB.py) and other related .py files
        - Main function of the beta-CROWN inspired ITNE global robustness certification algorithm.
        - Examples of using this beta-CROWN inspired ITNE algorithm can be found in [mnist_exp.py](mnist_exp.py) and [cifar_exp.py](cifar_exp.py).
    - [data/](data/)
        - Contains the MNIST and CIFAR10 models used for global robustness certification.
    - [ITNE_model/](ITNE_model/)
        - Globally-robust neural network training, which is a special DNN model based on Tensorflow that can be trained with the consideration of global robustness.
            - Using CROWN-based ITNE (not beta-crown based) to derive an differentiable global robustness metric, which can then be leveraged as a regularization term during training.
    - [net_repair/](net_repair/)
        - Contains all necessary code for NN global robustness repair.


## Improvement comparing to MILP-based ITNE
Output bound of CIFAR-10 network
| NN output | ITNE-MILP | ITNE-weakCROWN | improvement |
| --------- | --------- | -------------- | ----------- |
| 0  | [-1259, 1259] |    [-802, 846] |  34.6% |
| 1  | [-1336, 1336] |    [-888, 888] |  33.6% |
| 2  | [-1057, 1057] |    [-700, 664] |  35.5% |
| 3  | [-915, 915] |    [-607, 595] |  34.3% |
| 4  | [-1208, 1208] |    [-838, 816] |  31.5% |
|  5 | [-1138, 1138] |    [-768, 758] |  33.0% |
|  6 | [-1287, 1287] |    [-821, 848] |  35.2% |
|  7 | [-1241, 1241] |    [-867, 862] |  30.4% |
|  8 | [-1326, 1326] |    [-860, 881] |  34.4% |
|  9 | [-1266, 1266] |   [-840, 851] |  33.2% |

# Leveraging global robustness certification in adversarial training
The implementation can be find in folder [ITNE_model/](ITNE_model/).