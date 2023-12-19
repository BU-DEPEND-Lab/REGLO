import tensorflow as tf
import numpy as np

def cifar10_class_cube(target_class = 1):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_selected = np.stack([x_train[i] for i in range(len(y_train)) if y_train[i] == target_class])
    print(f'total number of images in class {target_class}: {len(x_selected)} and its shape is {x_selected.shape}.')
    class_lb = tf.math.reduce_min(x_selected, axis=0)
    class_ub = tf.math.reduce_max(x_selected, axis=0)
    feasible_dimension = tf.math.reduce_sum(tf.cast(tf.math.greater(class_ub - class_lb, tf.ones_like(class_lb)), tf.float32))
    # print(class_lb, class_ub)
    # print(feasible_dimension)
    return class_lb, class_ub

if __name__ == '__main__':
    # for i in range(10):
    #     print(cifar10_class_cube(i))
    class_lb, class_ub = cifar10_class_cube(0)
    print(class_lb.shape, class_ub.shape)