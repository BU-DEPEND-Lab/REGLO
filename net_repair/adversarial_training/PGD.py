import numpy as np
import tensorflow as tf


class ProjectedGradientDescent(object):
    """
    The Projected Gradient Descent attack is an iterative method in which,
    after each iteration, the perturbation is projected on an lp-ball of
    specified radius (in addition to clipping the values of the adversarial
    sample so that it lies in the permitted data range). This is the attack
    proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """
    def __init__(
        self,
        model,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        steps=20,
    ):
        """
        Create a class `ProjectedGradientDescent` instance for a given
        classifier.
        :param model: A given classifier to perform PGD attack.
        :param norm: The norm of the adversarial perturbation supporting
                     np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param steps: Total number of steps for PGD attack.
        """
        self.model = model
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.steps = steps

    def generate(self, x, log=True, single_idx=None):
        """
        Generate PGD perturbed input examples. Only support
        :param x: An array with the original inputs whose values are normalied
                  to [0, 1].
        :param log: Flag to print the log or not.
        :single_idx: PGD attack only on a single feature. The input feature
                     needs to be a vector.
        """
        if log:
            print("=============Generating {}-step PGD perturbed inputs with "
                  "eps {} and norm {}.=============".format(
                      self.steps, self.eps, self.norm
                  ))
        if single_idx is not None:
            assert (len(x.shape) == 2), "PGD attack on a single feature only supports vector inputs. But inputs x has shape {} here.".format(x.shapt)
            delta = np.zeros(x.shape)
            delta[:, single_idx] = (np.random.uniform(size=(
                x.shape[0],
            )) - 0.5) * 2 * self.eps
            delta = tf.convert_to_tensor(delta, dtype=tf.float32)
        else:
            delta = (tf.random.uniform(x.shape) - 0.5) * 2 * self.eps
        for _step in range(self.steps):
            with tf.GradientTape() as tape:
                tape.watch(delta)
                diff = self.model(x + delta) - self.model(x)
                loss = tf.reduce_mean(tf.abs(diff))

            # Get the gradients of the loss w.r.t. the inputs.
            gradient = tape.gradient(loss, delta)
            # Get the sign of the gradients to create the perturbation.
            if single_idx is not None:
                signed_grad = np.zeros(x.shape)
                signed_grad[:, single_idx] = tf.sign(
                    gradient
                )[:, single_idx].numpy()
                signed_grad = tf.convert_to_tensor(signed_grad,
                                                   dtype=tf.float32)
            else:
                signed_grad = tf.sign(gradient)
            # Perform a single-step perturbation.
            delta = delta + self.eps_step * self.eps * signed_grad
            delta = tf.clip_by_value(delta, -self.eps, self.eps)
        if log:
            print("=============PGD attack done.=============")
        return x + delta


class ProjectedGradientDescentForRegion(ProjectedGradientDescent):
    """
    Sample points from a given region and generate PGD perturbed inputs.
    """
    def __init__(
        self,
        model,
        input_lb,
        input_ub,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        steps=20,
    ):
        """
        Create a class `ProjectedGradientDescentForRegion` instance for a given
        classifier and a input region.
        :param model: A given classifier to perform PGD attack.
        :param input_lb: A lower bound of the input region.
        :param input_ub: A upper bound of the input region.
        :param norm: The norm of the adversarial perturbation supporting
                     np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param steps: Total number of steps for PGD attack.
        """
        super().__init__(model=model, norm=norm,
                         eps=eps, eps_step=eps_step, steps=steps)
        self.input_lb = input_lb
        self.input_ub = input_ub

    def random_sample(self, num_of_points=100, single_idx=None):
        """
        Randomly sample points from the input region and generate PGD perturbed
        inputs for training or evaluation.
        :param shape: The shape of inputs.
        :param num_of_points: A number of sampling points.
        """
        # Randomly sample points from the input region.
        input_diff = self.input_ub - self.input_lb
        random_x = input_diff * tf.random.uniform(
            (num_of_points, *self.input_lb.shape)) + self.input_lb

        # Generate PGD perturbed inputs on sampled points and ensure that the
        # perturbed inputs still lie inside the input region.
        perturbed_random_x = tf.clip_by_value(
            self.generate(random_x, log=False, single_idx=single_idx),
            self.input_lb, self.input_ub
        )

        return random_x, perturbed_random_x
