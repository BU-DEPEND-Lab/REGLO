# Leveraging global robustness certification in adversarial training

## MNIST example
The example code can be found in [mnist_ITNE.py](mnist_ITNE.py), where one can find how to create an ITNE model, evaluate global robustness, and train the model with global robustness regularization term.

When using Maxpooling layer after each convolution layer in this example, at least 9 GB GPU memory is needed.

The trained model is converted back to regular model and saved to [model/](model/). The trained model can then be evaluated by the [beta-crown inspired global robustness certification algorithm](../BnB.py). Evaluation code can be found in [eval_mnist.py](eval_mnist.py).

## CIFAR10 example
This example can be found in [cifar_ITNE.py](cifar_ITNE.py). Evaluation code can be found in [eval_cifar.py](eval_cifar.py).

## Note
For both training and evaluation, the commend to run the code is located at the end of each file. Note that the evaluation code need to run at the [root](../) directory.