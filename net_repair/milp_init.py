import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
# setting path
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from GlobalRobust_beta import nn_extractor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import model_from_json

import gurobipy as gp
from gurobipy import GRB
import time

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.savefig('mpg_training.png')

def save_model(model, fname):
    model_json = model.to_json()
    netname = 'model_auto_mpg_' + fname
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

def load_model(fname):
    netname = 'model_auto_mpg_' + fname
    json_filename = "./model/"+netname+".json"
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    dnn_model.load_weights("./model/"+netname+".h5")
    return dnn_model

def MPG_dataset():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    train_ds = tf.data.Dataset.from_tensor_slices(
        # (x_train, y_train)).batch(128)
        (train_features, train_labels)).shuffle(1000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(32)

    input_lb = np.array(train_features.describe().transpose()['min'])
    input_ub = np.array(train_features.describe().transpose()['max'])

    return train_ds, test_ds, normalizer, input_lb, input_ub

def save_mpg_bounds(input_lb, input_ub):
    save_path = "./mpg_bounds.pickle"
    with open(save_path, 'wb') as f:
        pickle.dump((input_lb, input_ub), f)

def load_mpg_bounds():
    save_path = "./mpg_bounds.pickle"
    with open(save_path, 'rb') as f:
        input_lb, input_ub = pickle.load(f)
    return input_lb, input_ub


def model_training():
    train_ds, test_ds, normalizer, input_lb, input_ub = MPG_dataset()

    dnn_model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    dnn_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam())

    history = dnn_model.fit(
        train_ds,
        verbose=0, 
        validation_data = test_ds,
        epochs=100)

    plot_loss(history)
    res = dnn_model.evaluate(test_ds, verbose=0)
    print(res)

    return dnn_model, input_lb, input_ub

def global_gradient(NN_infos, input_lb, input_ub, output_dim = 1, bigM=1024, dbg = False, timeout = None, max_points = 30, threshold=0, fairness_index = -1):
    """
    input_lb, input_ub: the lower and upper bound of the input states, which forms the input space
    output_dim: neural network output dimension
    bigM: Manually assign a bigM, which should be greater than any possible abosolute value of 
          the input of any relu node, and any intermidiate gradient. 
    dbg: whether to print gurobi optimization log
    timeout: A timeout for the gurobi solver
    fairness_index: feature index to evaluate fairness. -1 if evaluating global robustness.
    
    currently, only support one dimention input fully-connected NN
    
    return: the lower and upper bound of the global robustness f(x)-f(x')
    """
    gp_model = gp.Model('global_robustness')
    gp_model.Params.Threads = 8
    if not dbg:
        gp_model.Params.OutputFlag = 0
    if timeout is not None:
        gp_model.Params.timeLimit = timeout
    input_dim = len(input_lb)
    var_dict = {}
    x = gp_model.addVars(input_dim, lb=input_lb, ub = input_ub, vtype=GRB.CONTINUOUS, name='x_0')
    var_dict['x_0'] = x
    z_list = []
    jacobian = encode_NN(NN_infos, gp_model, var_dict, z_list, x, input_dim, bigM)

    abs_jacobian = gp_model.addVars(output_dim, input_dim, vtype=GRB.CONTINUOUS, name='abs_dx_out')
    gp_model.addConstrs((abs_jacobian[i, i_in] == gp.abs_(jacobian[i, i_in])
                            for i in range(output_dim) 
                            for i_in in range(input_dim)), name='abs_dx_out')
    row_sum = gp_model.addVars(output_dim, vtype=GRB.CONTINUOUS, name='abs_dx_row_sum')
    if fairness_index < 0:
        gp_model.addConstrs((row_sum[i] == abs_jacobian.sum(i, '*')
                                for i in range(output_dim)), name='abs_dx_row_sum')
    else:
        gp_model.addConstrs((row_sum[i] == abs_jacobian[i, fairness_index]
                                for i in range(output_dim)), name='abs_dx_row_sum')
    infy_norm = gp_model.addVar(vtype=GRB.CONTINUOUS, name='dx_out_infy_norm')
    gp_model.addConstr(infy_norm == gp.max_(row_sum), name='abs_dx_row_sum')
    gp_model.setObjective(infy_norm, GRB.MAXIMIZE)
    
    res = []
    x0s = []
    for _ in range(max_points):
        # print("Start to find worst-case gradient...", flush = True)
        t = time.time()
        gp_model.optimize()
        t = time.time() - t
        if gp_model.Status == GRB.OPTIMAL:
            res.append(gp_model.objVal)
            x0s.append([var_dict['x_0'][i].X for i in range(input_dim)])
            z_opt = {}
            abbr = 1
            for z_name in z_list:
                z_opt[z_name] = [var_dict[z_name][i].X > 0.5 for i in range(len(var_dict[z_name]))]
                abbr = binarize(abbr, z_opt[z_name])
            print(f'Greatest gradient {res[-1]:.4f}, activation pattern {bin(abbr)}, derived in {t: .2f}s.')
            if res[-1] < threshold:
                break
            rule_out_opt_activation(gp_model, var_dict, z_list, z_opt)
        elif gp_model.Status == GRB.TIME_LIMIT:
            print("Time limit exceeded for searching worst-case gradient, obj should between", 
                [gp_model.ObjBound, gp_model.objVal])
            res.append(gp_model.objVal)
            break
        else:
            print("Searching worst-case gradient: unknown status code", gp_model.Status)
            break
    
    return res, x0s

def encode_NN(NN_infos, gp_model, var_dict, z_list, x, in_shape, bigM):
    dx = np.diag(np.ones(in_shape))
    for k, layer_info in enumerate(NN_infos):
        x_prev = x
        dx_prev = dx
        if layer_info['type'] == 'normalization':
            mean = layer_info['mean']
            std = layer_info['std']
            n = layer_info['output_shape']
            x = gp_model.addVars(n, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_'+str(k+1))
            dx = gp_model.addVars(n, in_shape, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='dx_'+str(k+1))
            gp_model.addConstrs((x[i] == (x_prev[i] - mean[i])/std[i] 
                                 for i in range(n)), name='normalize_x_'+str(k))
            gp_model.addConstrs((dx[i, i_in] == (1/std[i]) * dx_prev[i, i_in] 
                                    for i in range(n) 
                                    for i_in in range(in_shape)), name=f'normalize_dx_{k}')
        elif layer_info['type'] == 'dense':
            W = layer_info['kernel']
            b = layer_info['bias']
            m = layer_info['input_shape']
            n = layer_info['output_shape']
            y = gp_model.addVars(n, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_'+str(k))
            dy = gp_model.addVars(n, in_shape, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='dy_'+str(k))
            gp_model.addConstrs((y[i] == gp.LinExpr(W[:, i], [x_prev[j] for j in range(m)]) + b[i]
                                    for i in range(n)), name='dense_x_'+str(k))
            gp_model.addConstrs((dy[i, i_in] == gp.LinExpr(W[:, i], [dx_prev[j, i_in] for j in range(m)])
                                    for i in range(n) 
                                    for i_in in range(in_shape)), name='dense_dx_'+str(k))
            
            var_dict['y_'+str(k)] = y
            var_dict['dy_'+str(k)] = dy
            x = gp_model.addVars(n, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_'+str(k+1))
            dx = gp_model.addVars(n, in_shape, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='dx_'+str(k+1))
            if layer_info['activation'] == 'relu': # Manually BigM
                z = gp_model.addVars(n, vtype=GRB.BINARY, name='z_'+str(k))
                # Indicator z: z=1 <=> y>=0, z=0 <=> y<=0
                var_dict['z_'+str(k)] = z
                z_list.append('z_'+str(k))
                gp_model.addConstrs((x[i] >= 0
                                        for i in range(n)), name='relu_bigM_1_x_'+str(k))
                gp_model.addConstrs((x[i] >= y[i]
                                        for i in range(n)), name='relu_bigM_2_x_'+str(k))
                gp_model.addConstrs((x[i] <= y[i] + bigM * (1-z[i])
                                        for i in range(n)), name='relu_bigM_3_x_'+str(k))
                gp_model.addConstrs((x[i] <= bigM * z[i]
                                        for i in range(n)), name='relu_bigM_4_x_'+str(k))

                gp_model.addConstrs((dx[i, i_in] <= bigM * z[i]
                                        for i in range(n) 
                                        for i_in in range(in_shape)), name='relu_bigM_5_x_'+str(k))
                gp_model.addConstrs((dx[i, i_in] >= -bigM * z[i]
                                        for i in range(n) 
                                        for i_in in range(in_shape)), name='relu_bigM_6_x_'+str(k))
                gp_model.addConstrs((dx[i, i_in] - dy[i, i_in] <= bigM * (1-z[i])
                                        for i in range(n) 
                                        for i_in in range(in_shape)), name='relu_bigM_7_x_'+str(k))
                gp_model.addConstrs((dx[i, i_in] - dy[i, i_in] >= -bigM * (1-z[i])
                                        for i in range(n) 
                                        for i_in in range(in_shape)), name='relu_bigM_8_x_'+str(k))
            elif layer_info['activation'] == 'linear':
                gp_model.addConstrs((x[i] == y[i]
                                     for i in range(n)), name='affine_x_'+str(k))
                gp_model.addConstrs((dx[i, i_in] == dy[i, i_in]
                                     for i in range(n) 
                                     for i_in in range(in_shape)), name='affine_dx_'+str(k))
            else:
                print("Error: Layer", k, "activation type not supported:", layer_info['activation'])
        else:
            print("Error: layer type not supported:", layer_info['type'])
        var_dict['x_'+str(k+1)] = x
        var_dict['dx_'+str(k+1)] = dx
    return var_dict['dx_'+str(len(NN_infos))]

def rule_out_opt_activation(gp_model, var_dict, z_list, z_opt):
    total_z = 0
    total_activated = 0
    sums = gp_model.addVars(len(z_list), vtype=GRB.CONTINUOUS, name='rule_out_sum_per_layer')
    for i, z_name in enumerate(z_list):
        n = len(var_dict[z_name])
        total_z += n
        activated = sum(z_opt[z_name])
        total_activated += activated
        coeff = [(-1)**(k+1) for k in z_opt[z_name]]    # if z_opt=0 (not activated), coeff=-1, otherwise coeff = 1
        gp_model.addConstr(sums[i] == gp.LinExpr(coeff, [var_dict[z_name][j] for j in range(n)]) + (n-activated), 
                            name=f'rule_out_sum_{z_name}')
    gp_model.addConstr(sums.sum() <= total_z - 1, name=f'rule_out_total')

def binarize(abbr: int, z_opt):
    for k in z_opt:
        abbr = (abbr << 1) + k
    return abbr
    

if __name__ == '__main__':
    # model, input_lb, input_ub = model_training()
    # save_model(model, '20x20')
    # save_mpg_bounds(input_lb, input_ub)
    model = load_model('20x20')
    input_lb, input_ub = load_mpg_bounds()
    t = time.time()
    Layers = nn_extractor(model)
    wc_gradient, input_samples = global_gradient(Layers, input_lb, input_ub, dbg=False, timeout=15000, max_points=5) 
    t = time.time() - t
    print(wc_gradient)
    print(np.array(input_samples).shape)
    print("time consuming:", t)
    t = time.time()
    wc_gradient, input_samples = global_gradient(Layers, input_lb, input_ub, dbg=False, timeout=15000, max_points=5, fairness_index=6) 
    t = time.time() - t
    print(wc_gradient)
    print(np.array(input_samples).shape)
    print("time consuming:", t)