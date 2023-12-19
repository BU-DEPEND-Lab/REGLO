import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
# setting path
sys.path.append('../')
from GlobalRobust_beta import nn_extractor
import pickle

from enum import Enum, auto
from collections import deque
import tensorflow as tf
import numpy as np
import time

from Model_ITNE import Model_ITNE
from Relu_layer_ITNE import Relu_ITNE
from Conv2d_layer_ITNE import Conv2d_ITNE
from Flatten_layer_ITNE import Flatten_ITNE
from Linear_layer_ITNE import Linear_ITNE

def creat_mnist_model(use_maxpool = True):
    model = Model_ITNE()
    if use_maxpool:
        layer_id = model.add_layer(Conv2d_ITNE(16, 3))
        layer_id = model.add_maxpool_layer(layer_id)
        layer_id = model.add_layer(Relu_ITNE(), input_ids=[layer_id])
        layer_id = model.add_layer(Conv2d_ITNE(32, 3), input_ids=[layer_id])
        layer_id = model.add_maxpool_layer(layer_id)
    else:
        layer_id = model.add_layer(Conv2d_ITNE(16, 3, strides=2))
        layer_id = model.add_layer(Relu_ITNE(), input_ids=[layer_id])
        layer_id = model.add_layer(Conv2d_ITNE(32, 3, strides=2), input_ids=[layer_id])

    layer_id = model.add_layer(Relu_ITNE(), input_ids=[layer_id])
    layer_id = model.add_layer(Flatten_ITNE(), input_ids=[layer_id])
    layer_id = model.add_layer(Linear_ITNE(16), input_ids=[layer_id])
    layer_id = model.add_layer(Relu_ITNE(), input_ids=[layer_id])
    layer_id = model.add_layer(Linear_ITNE(10), input_ids=[layer_id])

    print("total layers", layer_id)

    model.update_outbounds()

    # inputs = tf.zeros((1,28,28,1))
    # ouput = model(inputs)
    input_shape = (28,28,1)
    model.update_shapes(input_shape)
    dist_bound = (2/255) * tf.ones(input_shape)
    model.set_input_bounds(-dist_bound, dist_bound)

    # for layer in model.layers:
    #     print(layer.name)

    return model

def duplicate_sequential_model(itne_model):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, 2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    inputs = tf.random.uniform(shape=(16,28,28,1))

    i = 0
    j = 0
    while i < len(itne_model.layers) and j < len(model.layers):
        if 'flatten' in model.layers[j].name:
            j += 1
        elif 'flatten' in itne_model.layers[i].name:
            i += 1
        elif 'relu' in itne_model.layers[i].name:
            i += 1
        elif 'conv2d' in itne_model.layers[i].name or 'linear' in itne_model.layers[i].name:
            print("Copying", itne_model.layers[i].name, "layer weights to", model.layers[j].name)
            model.layers[j].set_weights(itne_model.layers[i].get_weights())
            i += 1
            j += 1
        else:
            print("Unknown layer type:", itne_model.layers[i].name, model.layers[j].name)
            break

    ouput = model(inputs)
    output2 = itne_model(inputs)
    assert tf.reduce_max(tf.abs(output2 - ouput)).numpy() < 1e-6
    
    return model

def save_model(model, fname):
    model_json = model.to_json()
    netname = 'model_itne_mnist_' + fname
    with open("./model/"+netname+".json", 'w') as json_file:
        json_file.write(model_json)
    model.save_weights("./model/"+netname+".h5")
    print("Saved model to disk")
    extract_and_save(model, netname)

def extract_and_save(model, netname):
    Layers = nn_extractor(model)
    pickle_filename = 'model/' + netname + '.pickle'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(Layers, f)
    print("Model infos are extracted and saved.")

def model_training(model):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(1024)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1024)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    global_robustness = tf.keras.metrics.Mean(name='global_robustness')

    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            output_variation = tf.reduce_mean(model.output_variation())
            loss = loss_object(labels, predictions) + 0.05 * output_variation
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        global_robustness(output_variation)

    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 100

    output_variation = model.output_variation()
    print(f'Initial global robustness', output_variation.numpy())

    for epoch in range(EPOCHS):

        t = time.time()

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        global_robustness.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result(): .3f}, '
            f'Accuracy: {train_accuracy.result() * 100: .2f}, '
            f'Test Loss: {test_loss.result(): .3f}, '
            f'Test Accuracy: {test_accuracy.result() * 100: .2f}, '
            f'Average Robustness: {global_robustness.result(): .2f}, '
            f'Training time: {time.time() - t: .2f}s',
            flush=True
        )

    output_variation = model.output_variation()
    print(f'Final global robustness', output_variation.numpy())

    return model



if __name__ == '__main__':

    model = creat_mnist_model(use_maxpool=False)

    model_training(model)
    tf_model = duplicate_sequential_model(model)
    save_model(tf_model, '2c2d')



"""
    nohup python mnist_ITNE.py > results/train_log_mnist_2c2d.log 2>&1 &
"""