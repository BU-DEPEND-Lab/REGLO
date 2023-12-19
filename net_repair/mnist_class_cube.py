import tensorflow as tf
import numpy as np

def mnist_class_cube(target_class = 1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_selected = np.stack([x_train[i] for i in range(len(y_train)) if y_train[i] == target_class])
    # print(f'total number of images in class {target_class}: {len(x_selected)}.')
    class_lb = tf.math.reduce_min(x_selected, axis=0)
    class_ub = tf.math.reduce_max(x_selected, axis=0)
    feasible_dimension = tf.math.reduce_sum(tf.cast(tf.math.greater(class_ub - class_lb, tf.ones_like(class_lb)), tf.float32))
    # print(class_lb, class_ub)
    # print(feasible_dimension)
    return class_lb, class_ub

if __name__ == '__main__':
    for i in range(10):
        mnist_class_cube(i)