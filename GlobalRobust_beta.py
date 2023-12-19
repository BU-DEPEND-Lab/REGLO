import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import pickle
import time

def nn_extractor(dnn_model):
    Layers = []
    for layer in dnn_model.layers:
        layer_info = {}
        if 'batch_normalization' in layer.name:
            # print(dir(layer))
            layer_info['type'] = 'normalization'
            layer_info['input_shape'] = layer.input_shape[1:]
            layer_info['output_shape'] = layer.output_shape[1:]
            layer_info['dim'] = len(layer_info['output_shape'])
            if layer_info['dim'] == 1:
                layer_info['input_shape'] = layer_info['input_shape'][0]
                layer_info['output_shape'] = layer_info['output_shape'][0]
            mu = layer.moving_mean.numpy()
            sigma = np.sqrt(layer.moving_variance.numpy() + layer.epsilon)
            # sigma = layer.moving_variance.numpy() + layer.epsilon
            beta = layer.beta.numpy()
            gamma = layer.gamma.numpy()
            std = sigma/gamma
            mean = mu - std * beta
            layer_info['mean'] = np.ones(layer_info['output_shape']) * mean
            layer_info['std'] = np.ones(layer_info['output_shape']) * std
            # print(std.shape, mean.shape, layer_info['mean'].shape, layer_info['std'].shape)
            # if layer_info['dim'] == 1:
            #     print(layer_info['mean'][:5], mean[:5])
            print(f"batch normalization in {layer_info['dim']} dimension with shape {layer_info['output_shape']}")
        elif 'normalization' in layer.name:
            # print(dir(layer))
            layer_info['type'] = 'normalization'
            layer_info['mean'] = layer.mean.numpy()
            layer_info['std'] = np.sqrt(layer.variance.numpy())
            layer_info['dim'] = len(layer_info['mean'].shape)
            if layer_info['dim'] == 1:
                layer_info['input_shape'] = layer_info['mean'].shape[0]
                layer_info['output_shape'] = layer_info['mean'].shape[0]
            else:
                layer_info['input_shape'] = layer_info['mean'].shape
                layer_info['output_shape'] = layer_info['mean'].shape
            # print('normalization, mean shape:', layer_info['mean'].shape)
            print('normalization, std shape:', layer_info['std'].shape)
        elif 'dense' in layer.name:
            layer_info['type'] = 'dense'
            layer_info['kernel'] = layer.kernel.numpy()
            layer_info['bias'] = layer.bias.numpy()
            config = layer.get_config()
            layer_info['activation'] = config['activation']
            layer_info['input_shape'] = layer_info['kernel'].shape[0]
            layer_info['output_shape'] = layer_info['kernel'].shape[1]
            # print('dense kernel shape:', layer_info['kernel'].shape)
            # print('dense bias shape:', layer_info['bias'].shape)
            print('dense activation:', layer_info['activation'])
        elif 'conv2d' in layer.name:
            layer_info['type'] = 'conv2d'
            layer_info['kernel'] = layer.kernel.numpy()
            layer_info['bias'] = layer.bias.numpy()
            layer_info['input_shape'] = layer.input_shape[1:]   # Remove the first dimension (i.e., batch size)
            layer_info['output_shape'] = layer.output_shape[1:]
            config = layer.get_config()
            layer_info['kernel_size'] = config['kernel_size']
            layer_info['strides'] = config['strides']
            layer_info['padding'] = config['padding']
            layer_info['activation'] = config['activation']
            layer_info['flatten_output'] = False
            print(config['name'], layer_info['input_shape'], layer_info['output_shape'], layer_info['activation'])
            print(layer_info['kernel'].shape)
            if config['data_format'] != 'channels_last':
                raise Exception("Conv2d: data format not supported yet: " + config['data_format'])
            if config['dilation_rate'][0] != 1 or config['dilation_rate'][1] != 1:
                raise Exception("Conv2d: dilation not supported yet: " + config['dilation_rate'])
        elif 'flatten' in layer.name:
            last_layer_info = Layers[-1]
            if last_layer_info['type'] == 'normalization':
                last_layer_info = Layers[-2]
            if last_layer_info['type'] != 'conv2d':
                raise Exception("Flatten: previous layer is not Conv2d: " + last_layer_info['type'])
            last_layer_info['flatten_output'] = True
        elif 'dropout' in layer.name:
            pass
        else:
            raise Exception("layer type not supported: "+layer.name)
        if layer_info:
            Layers.append(layer_info)
    return Layers

class RP_GlobR:
    def __init__(self, NN_infos, dbg = False, bigM = None, timeout = None, lp_relax = False, nThreads = None):
        """
        dbg: whether to print gurobi optimization log
        bigM: Manually assign a bigM, which should be greater than any possible abosolute value of 
              the input of any relu node. Default is None, where the general max constraint is used
              instead of BigM (which will be inplicitly converted to BigM by Gurobi.)
        timeout: Set timeout for gurobi optimization, if the optimizer doesn't find optimal solution
              after timeout, the objective bound will be chosen as an overapproximation
        lp_relax: Flag to choose whether to relax "relu(x+delta) - relu(x)" by linear constraints
        """
        self.NN_infos = NN_infos
        self.dbg = dbg
        self.bigM = bigM
        self.timeout = timeout
        self.lp_relax = lp_relax
        self.preReluRange = {}
        self.map_layer_ids()
        self.refine_config = {}
        self.nThreads = nThreads

    def map_layer_ids(self):
        self.relu_ids = [-1]
        for k, layer_info in enumerate(self.NN_infos):
            if 'activation' in layer_info and layer_info['activation'] == 'relu':
                self.relu_ids.append(k)
        if self.relu_ids[-1] < len(self.NN_infos) - 1:
            self.relu_ids.append(len(self.NN_infos) - 1)

    def encode_one_node(
        self, gp_model, var_dict, relu_layer_id, node_id, suffix = '', window_size = 1
    ):
        out_layer_id = self.relu_ids[relu_layer_id]
        # out_layer = self.NN_infos[out_layer_id]
        x_out = gp_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_'+str(out_layer_id+1)+suffix)
        y_out = gp_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_'+str(out_layer_id)+suffix)
        x_in = {node_id: x_out}
        var_dict['x_'+str(out_layer_id + 1)+suffix] = x_in
        in_ids = [node_id]
        start_layer = self.relu_ids[max(0, relu_layer_id - window_size)] + 1
        for k in reversed(range(start_layer, out_layer_id + 1)):
            layer_info = self.NN_infos[k]
            out_ids = in_ids
            x = x_in
            if layer_info['type'] == 'normalization':
                # pass
                mean = layer_info['mean']
                std = layer_info['std']
                if layer_info['dim'] == 1:
                    n = layer_info['output_shape']
                    _mean = lambda i: mean[i]
                    _std = lambda i: std[i]
                elif layer_info['dim'] != 3:
                    print("Error: normalization only supported for dimension 1 and 3")
                    raise Exception()
                else:
                    wout, hout, cout = layer_info['output_shape']
                    n = wout*hout*cout
                    _mean = lambda i: mean[i//(hout*cout), (i//cout)%hout, i%cout]
                    _std = lambda i: std[i//(hout*cout), (i//cout)%hout, i%cout]
                in_ids = out_ids
                x_in = gp_model.addVars(in_ids, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_'+str(k)+suffix)
                gp_model.addConstrs((x[i] == (x_in[i] - _mean(i))/_std(i)
                                     for i in in_ids), 
                                    name='normalize_x_'+str(k)+suffix)
            # elif layer_info['type'] == 'dense':
            elif layer_info['type'] in ['dense', 'conv2d']:
                W = layer_info['kernel']
                b = layer_info['bias']
                if layer_info['type'] == 'dense':
                    m = layer_info['input_shape']
                    in_ids = list(range(m))
                    n = layer_info['output_shape']
                elif layer_info['type'] == 'conv2d':
                    win, hin, cin = layer_info['input_shape']
                    wout, hout, cout = layer_info['output_shape']
                    m = win * hin * cin
                    n = wout * hout * cout
                    strides = layer_info['strides']
                    padding = layer_info['padding'] #TODO: currently only the default one is supported
                    knsz = layer_info['kernel_size']
                    
                    _inid = lambda i1, i2, i3: i1*hin*cin + i2*cin + i3
                    _outid = lambda i1, i2, i3: i1*hout*cout + i2*cout + i3
                    _Wls = lambda i4: [W[i1, i2, i3, i4] for i1 in range(knsz[0])
                                                         for i2 in range(knsz[1])
                                                         for i3 in range(cin)]
                    in_ids = {
                        _inid(strides[0] * j1 + i1, strides[1] * j2 + i2, i3)
                        for i1 in range(knsz[0])
                        for i2 in range(knsz[1])
                        for i3 in range(cin)
                        for j1 in range(wout)
                        for j2 in range(hout)
                        for j3 in range(cout)
                        if _outid(j1, j2, j3) in out_ids
                    }
                    in_ids = list(in_ids)
                x_in = gp_model.addVars(in_ids, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='x_'+str(k)+suffix)
                if k == out_layer_id:
                    y = {node_id: y_out}
                else:
                    y = gp_model.addVars(out_ids, lb=-float('inf'), vtype=GRB.CONTINUOUS, name='y_'+str(k)+suffix)
                var_dict['y_'+str(k)+suffix] = y
                if layer_info['type'] == 'dense':
                    gp_model.addConstrs((y[i] == gp.LinExpr(W[:, i], [x_in[j] for j in range(m)]) + b[i]
                                        for i in out_ids), name='dense_x_'+str(k)+suffix)
                elif layer_info['type'] == 'conv2d':
                    _xls = lambda j1, j2: [
                        x_in[_inid(strides[0] * j1 + i1, strides[1] * j2 + i2, i3)] 
                        for i1 in range(knsz[0])
                        for i2 in range(knsz[1])
                        for i3 in range(cin)]
                    gp_model.addConstrs(
                        (
                            y[_outid(j1, j2, j3)] == b[j3] + gp.LinExpr(_Wls(j3), _xls(j1, j2))
                            for j1 in range(wout)
                            for j2 in range(hout)
                            for j3 in range(cout)
                            if _outid(j1, j2, j3) in out_ids
                        ),
                        name='conv2d_x_'+str(k)+suffix
                    )
                if k == out_layer_id:       # Beta update: no constraints between y_out and x_out in current encoding
                    var_dict['x_'+str(k)+suffix] = x_in
                    continue
                if layer_info['activation'] == 'relu':
                    has_bound = False
                    if 'y_'+str(k) in self.preReluRange and k < out_layer_id:
                        has_bound = True
                        if suffix == '':
                            y_lb = self.preReluRange['y_'+str(k)]['lb']
                            y_ub = self.preReluRange['y_'+str(k)]['ub']
                            x_lb = self.preReluRange['y_'+str(k)]['x_lb']
                            x_ub = self.preReluRange['y_'+str(k)]['x_ub']
                        else:
                            # eps = 1e-8
                            y_orig = var_dict['y_'+str(k)]
                            x_orig = var_dict['x_'+str(k+1)]
                            dy_lb = self.preReluRange['y_'+str(k)]['diff_lb']
                            dy_ub = self.preReluRange['y_'+str(k)]['diff_ub']
                            dx_lb = self.preReluRange['y_'+str(k)]['x_diff_lb']
                            dx_ub = self.preReluRange['y_'+str(k)]['x_diff_ub']
                    if has_bound:
                        for i in out_ids:
                            if suffix == '':
                                y[i].lb = y_lb[i]
                                y[i].ub = y_ub[i]
                                x[i].lb = x_lb[i]
                                x[i].ub = x_ub[i]
                            else:
                                gp_model.addConstr(y[i] - y_orig[i] >= dy_lb[i], 
                                                    name='relu_dy_lb_'+str(k) +'_'+str(i))
                                gp_model.addConstr(y[i] - y_orig[i] <= dy_ub[i], 
                                                    name='relu_dy_ub_'+str(k) +'_'+str(i))
                                gp_model.addConstr(x[i] - x_orig[i] >= dx_lb[i], 
                                                    name='relu_dx_lb_'+str(k) +'_'+str(i))
                                gp_model.addConstr(x[i] - x_orig[i] <= dx_ub[i], 
                                                    name='relu_dx_ub_'+str(k) +'_'+str(i))
                    relax_nodes = []
                    if self.lp_relax and k < out_layer_id:
                        if not has_bound:
                            raise Exception("For LP relaxiation, the range of each relu input needs to be set.\
                                            Check the code, the range of", 'y_'+str(k), "is not in the dict, \
                                            preReluRange.")
                        
                        if window_size in self.refine_config:
                            rconf = self.refine_config[window_size]
                            if suffix == '':
                                n_to_refine = math.ceil(rconf['p_orig'] * len(out_ids))
                                n_to_refine = min(n_to_refine, rconf['max_orig'])
                                rconf['set_orig'].add(n_to_refine)
                                refine_ids = [i for i in out_ids if y_lb[i] < 0 and y_ub[i] > 0]
                                scores = [-y_lb[i]*y_ub[i]/(y_ub[i] - y_lb[i]) for i in refine_ids]
                            else:
                                n_to_refine = math.ceil(rconf['p_diff'] * len(out_ids))
                                n_to_refine = min(n_to_refine, rconf['max_diff'])
                                rconf['set_diff'].add(n_to_refine)
                                refine_ids = out_ids
                                scores = [max(abs(dy_lb[i]), abs(dy_ub[i])) for i in refine_ids]
                            sorted_idx = np.argsort(scores)
                            refine_ids = [refine_ids[j] for i, j in enumerate(reversed(sorted_idx)) if i < n_to_refine]
                            # np.random.shuffle(refine_ids)
                            # refine_ids = [refine_ids[i] for i in range(len(refine_ids)) if i < n_to_refine]
                            # rconf['refined_index'] = refine_ids
                            relax_nodes = [i for i in out_ids if i not in refine_ids]
                        else:
                            relax_nodes = [i for i in out_ids]
                    for i in relax_nodes:
                        if suffix == '':
                            lbi = min(y_lb[i], 0)
                            ubi = max(y_ub[i], 0)
                            diff = ubi - lbi if ubi - lbi > 0 else 1
                            gp_model.addConstr(x[i] >= 0, 
                                            name='relu_lpRelax_1_x_'+str(k) +'_'+str(i))
                            gp_model.addConstr(x[i] >= y[i], 
                                            name='relu_lpRelax_2_x_'+str(k) +'_'+str(i))
                            gp_model.addConstr(x[i] <= ubi * (y[i] - lbi)/diff, 
                                            name='relu_lpRelax_3_x_'+str(k) +'_'+str(i))
                        else:
                            lbi = min(dy_lb[i], 0)
                            ubi = max(dy_ub[i], 0)
                            diff = (ubi - lbi) if ubi - lbi > 0 else 1
                            gp_model.addConstr(x[i] - x_orig[i] >= lbi * (ubi - (y[i] - y_orig[i]))/diff, 
                                            name='relu_lpRelax_1_dx_'+str(k) +'_'+str(i))
                            gp_model.addConstr(x[i] - x_orig[i] <= ubi * (y[i] - y_orig[i] - lbi)/diff, 
                                                name='relu_lpRelax_2_dx_'+str(k) +'_'+str(i))
                    for i in out_ids:
                        if i in relax_nodes:
                            continue
                        if self.bigM is None:
                            if suffix == '' or not has_bound: # has_bound is equivalent to x_orig is valid when suffix != 0
                                gp_model.addConstr(x[i] == gp.max_(y[i], 0), name='relu_x_'+str(k)+suffix+str(i))
                            else:
                                gp_model.addConstr(x[i] == gp.max_(y[i], 0), name='relu_x_'+str(k)+suffix+str(i))
                                gp_model.addConstr(x_orig[i] == gp.max_(y_orig[i], 0), name='relu_x_'+str(k)+str(i))
                        else: # Manually BigM
                            z = gp_model.addVar(vtype=GRB.BINARY, name='z_'+str(k)+suffix+str(i))
                            # Indicator z: z=1 <=> y>=0, z=0 <=> y<=0
                            var_dict['z_'+str(k)+suffix+str(i)] = z
                            gp_model.addConstr(x[i] >= 0, name='relu_bigM_1_x_'+str(k)+suffix+str(i))
                            gp_model.addConstr(x[i] >= y[i], name='relu_bigM_2_x_'+str(k)+suffix+str(i))
                            gp_model.addConstr(x[i] <= y[i] + self.bigM * (1-z), name='relu_bigM_3_x_'+str(k)+suffix+str(i))
                            gp_model.addConstr(x[i] <= self.bigM * z, name='relu_bigM_4_x_'+str(k)+suffix+str(i))
                            if suffix == '' or not has_bound:
                                pass
                            elif 'z_'+str(k)+str(i) not in var_dict:
                                # For bigM, only implemented refinment method 1 (sperate-refinement)
                                z_orig = gp_model.addVar(vtype=GRB.BINARY, name='z_'+str(k)+str(i))
                                # Indicator z: z=1 <=> y>=0, z=0 <=> y<=0
                                var_dict['z_'+str(k)+str(i)] = z_orig
                                gp_model.addConstr(x_orig[i] >= 0, name='relu_bigM_1_x_'+str(k)+str(i))
                                gp_model.addConstr(x_orig[i] >= y_orig[i], name='relu_bigM_2_x_'+str(k)+str(i))
                                gp_model.addConstr(x_orig[i] <= y_orig[i] + self.bigM * (1-z_orig), name='relu_bigM_3_x_'+str(k)+str(i))
                                gp_model.addConstr(x_orig[i] <= self.bigM * z_orig, name='relu_bigM_4_x_'+str(k)+str(i))

                elif layer_info['activation'] == 'linear':
                    gp_model.addConstrs((x[i] == y[i]
                                         for i in out_ids), name='affine_x_'+str(k)+suffix)
                else:
                    print("Error: Layer", i, "activation type not supported:", layer_info['activation'])
            else:
                print("Error: layer type not supported:", layer_info['type'])
            var_dict['x_'+str(k)+suffix] = x_in
        return x_in, in_ids, x_out, y_out

    def encoding_relu_output_constraint(self, gp_model,
                                   x_out, x_d_out, y_out, y_d_out, 
                                   y_lb, y_ub, dy_lb, dy_ub,
                                   lp_relax = True):
        if lp_relax:
            y_out.lb = y_lb
            y_out.ub = y_ub
            gp_model.addConstr(y_d_out - y_out >= dy_lb, 
                                name='relu_dy_lb_out')
            gp_model.addConstr(y_d_out - y_out <= dy_ub, 
                                name='relu_dy_ub_out')
            lbi = min(y_lb, 0)
            ubi = max(y_ub, 0)
            diff = ubi - lbi if ubi - lbi > 0 else 1
            gp_model.addConstr(x_out >= 0, 
                            name='relu_lpRelax_1_x_out')
            gp_model.addConstr(x_out >= y_out, 
                            name='relu_lpRelax_2_x_out')
            gp_model.addConstr(x_out <= ubi * (y_out - lbi)/diff, 
                            name='relu_lpRelax_3_x_out')
            lbi = min(dy_lb, 0)
            ubi = max(dy_ub, 0)
            diff = (ubi - lbi) if ubi - lbi > 0 else 1
            gp_model.addConstr(x_d_out - x_out >= lbi * (ubi - (y_d_out - y_out))/diff, 
                            name='relu_lpRelax_1_dx_out')
            gp_model.addConstr(x_d_out - x_out <= ubi * (y_d_out - y_out - lbi)/diff, 
                                name='relu_lpRelax_2_dx_out')
        else:
            if self.bigM is None:
                gp_model.addConstr(x_d_out == gp.max_(y_d_out, 0), name='relu_x_d_out')
                gp_model.addConstr(x_out == gp.max_(y_out, 0), name='relu_x_out')
            else: # Manually BigM
                z = gp_model.addVar(vtype=GRB.BINARY, name='z_out')
                # Indicator z: z=1 <=> y>=0, z=0 <=> y<=0
                # var_dict['z_out'] = z
                gp_model.addConstr(x_out >= 0, name='relu_bigM_1_x_out')
                gp_model.addConstr(x_out >= y_out, name='relu_bigM_2_x_out')
                gp_model.addConstr(x_out <= y_out + self.bigM * (1-z), name='relu_bigM_3_x_out')
                gp_model.addConstr(x_out <= self.bigM * z, name='relu_bigM_4_x_out')
                z_d = gp_model.addVar(vtype=GRB.BINARY, name='z_d_out')
                # var_dict['z_d_out'] = z_d
                gp_model.addConstr(x_d_out >= 0, name='relu_bigM_1_x_d_out')
                gp_model.addConstr(x_d_out >= y_d_out, name='relu_bigM_2_x_d_out')
                gp_model.addConstr(x_d_out <= y_d_out + self.bigM * (1-z_d), name='relu_bigM_3_x_d_out')
                gp_model.addConstr(x_d_out <= self.bigM * z_d, name='relu_bigM_4_x_d_out')


    
    def gp_optimize_3(self, gp_model, find_lb, objective, name = " "):
        """
        find_lb: True if looking for lower bound, False if looking for upper bound
        """
        def _optimize(gp_model, obj_name):
            if self.dbg:
                print("Start to find", obj_name, "bound...", flush = True)
            gp_model.optimize()
            if gp_model.Status == GRB.OPTIMAL:
                # obj = gp_model.objVal
                obj = gp_model.ObjBound
            elif gp_model.Status == GRB.TIME_LIMIT:
                print("Time limit exceeded for", obj_name, "bound optimization, obj should between", 
                    [gp_model.ObjBound, gp_model.objVal])
                obj = gp_model.ObjBound
            else:
                print(obj_name, "bound optimization: unknown status code", gp_model.Status)
            return obj

        if find_lb:
            bound_name = "lower"
            grb_obj = GRB.MINIMIZE
        else:
            bound_name = "upper"
            grb_obj = GRB.MAXIMIZE

        gp_model.setObjective(objective, grb_obj)
        bound = _optimize(gp_model, name + " " + bound_name)
        return bound
        
        

    def range_prop_window(self, input_lb, input_ub, diff_lb, diff_ub, relu_layer_id, window_size = 1, updateRange = False):
        """
        Note: unlike range_prop_layer(), where the relu_layer_id is the input layer, here relu_layer_id is the id of the last
              layer
        For refinement of some relexed nodes, please call set_refine_config() first
        
        updateRange: whether update the output range of the window. When it is True, we assume that this function 
                     is called layer by layer rather than window by window.  
        """
        out_layer_id = self.relu_ids[relu_layer_id]
        start_layer = self.relu_ids[max(0, relu_layer_id - window_size)] + 1
        out_layer_size = np.prod(self.NN_infos[out_layer_id]['output_shape'])

        _d_suffix = '_d'

        layer_info = self.NN_infos[out_layer_id]
        if 'activation' not in layer_info:
            raise Exception("Window output layer is not Dense/Conv, but " + layer_info['type'])
        out_layer_activation = layer_info['activation']

        out_layer_relax = True
        if window_size in self.refine_config:
            rconf = self.refine_config[window_size]
            if rconf['max_diff'] > 0 and rconf['p_diff'] > 0:
                out_layer_relax = False
        
        preReluRange_needUpdate = updateRange
        if self.lp_relax:
            for k in range(start_layer, out_layer_id + 1):
                layer_info = self.NN_infos[k]
                if 'activation' in layer_info and layer_info['activation'] == 'relu' and 'y_'+str(k) not in self.preReluRange:
                    preReluRange_needUpdate = True
                    break
        if preReluRange_needUpdate:
            if window_size > 1 and not updateRange and out_layer_id < len(self.NN_infos) - 1:
                raise Exception("For LP relaxiation, window_size needs to be firstly set to 1 \
                                to first initialize the range of relu input.")
            if window_size > 1 and not updateRange and out_layer_id == len(self.NN_infos) - 1:
                preReluRange_needUpdate = False
            y_lb = []
            y_ub = []
            y_diff_lb = []
            y_diff_ub = []

        output_lb = []
        output_ub = []
        out_diff_lb = []
        out_diff_ub = []

        print("Relu layer", relu_layer_id, "total neurons", out_layer_size, flush=True)
        for i in range(out_layer_size):
            time_cur = time.time()
            # if out_layer_size > 10 and i%(out_layer_size//5) > 0:
            if time_cur < self.time_last_print + 120:
                pass
            else:
                self.time_last_print = time_cur
                if preReluRange_needUpdate:
                    print(f"optimize the input+output range of relu layer {relu_layer_id} node {i} at {(self.time_last_print - self.time_0)//60} minutes", flush=True)
                else:
                    print(f"optimize the output range of relu layer {relu_layer_id} node {i} at {(self.time_last_print - self.time_0)//60} minutes", flush=True)
            
            gp_model = gp.Model('global_robustness')
            if not self.dbg:
                gp_model.Params.OutputFlag = 0
            if self.timeout is not None:
                gp_model.Params.timeLimit = self.timeout / out_layer_size
            if self.nThreads is not None:
                gp_model.Params.Threads = self.nThreads
            var_dict = {}
            x, in_ids, x_out, y_out = self.encode_one_node(gp_model, var_dict, relu_layer_id, i, window_size=window_size)
            for j in in_ids:
                x[j].lb = input_lb[j]
                x[j].ub = input_ub[j]
            
            y_l = self.gp_optimize_3(gp_model, True, y_out)
            y_u = self.gp_optimize_3(gp_model, False, y_out)
            if out_layer_activation == 'relu':
                out_l = max(y_l, 0)
                out_u = max(y_u, 0)
            elif out_layer_activation == 'linear':
                out_l = y_l
                out_u = y_u
            else:
                raise Exception(f"Error: Layer {out_layer_id} activation type not supported: {layer_info['activation']}")

            if preReluRange_needUpdate:
                y_lb.append(y_l)
                y_ub.append(y_u)
            
            x_d, _, x_d_out, y_d_out = self.encode_one_node(gp_model, var_dict, relu_layer_id, i, _d_suffix, window_size=window_size)
            delta = gp_model.addVars(
                in_ids, 
                lb = [diff_lb[j] for j in in_ids], 
                ub = [diff_ub[j] for j in in_ids], 
                vtype=GRB.CONTINUOUS, name='delta'
            )
            gp_model.addConstrs((x_d[j] == x[j] + delta[j] for j in in_ids), name='disturbance_constr')
            var_dict['delta'] = delta

            if out_layer_activation == 'linear' or (out_layer_activation == 'relu' and out_layer_relax) or preReluRange_needUpdate:
                y_diff_l = self.gp_optimize_3(gp_model, True, (y_d_out - y_out))
                y_diff_u = self.gp_optimize_3(gp_model, False, (y_d_out - y_out))
            else:
                y_diff_l = y_diff_u = None

            if preReluRange_needUpdate:
                y_diff_lb.append(y_diff_l)
                y_diff_ub.append(y_diff_u)
            
            if out_layer_activation == 'linear':
                diff_l, diff_u = y_diff_l, y_diff_u
            elif out_layer_activation == 'relu':
                self.encoding_relu_output_constraint(gp_model, x_out, x_d_out, y_out, y_d_out, y_l, y_u, y_diff_l, y_diff_u, lp_relax=out_layer_relax)
                diff_l = self.gp_optimize_3(gp_model, True, (x_d_out - x_out))
                diff_u = self.gp_optimize_3(gp_model, False, (x_d_out - x_out))
            else:
                raise Exception(f"Error: Layer {out_layer_id} activation type not supported: {layer_info['activation']}")

            output_lb.append(out_l)
            out_diff_lb.append(diff_l)
            output_ub.append(out_u)
            out_diff_ub.append(diff_u)

        if preReluRange_needUpdate:
            y_name = 'y_'+str(out_layer_id)
            if y_name in self.preReluRange:
                self.preReluRange[y_name]['lb'] = [max(i,j) for i, j in zip(y_lb, self.preReluRange[y_name]['lb'])]
                self.preReluRange[y_name]['ub'] = [min(i,j) for i, j in zip(y_ub, self.preReluRange[y_name]['ub'])]
                self.preReluRange[y_name]['diff_lb'] = [max(i,j) for i, j in zip(y_diff_lb, self.preReluRange[y_name]['diff_lb'])]
                self.preReluRange[y_name]['diff_ub'] = [min(i,j) for i, j in zip(y_diff_ub, self.preReluRange[y_name]['diff_ub'])]

                self.preReluRange[y_name]['x_lb'] = [max(i,j) for i, j in zip(output_lb, self.preReluRange[y_name]['x_lb'])]
                self.preReluRange[y_name]['x_ub'] = [min(i,j) for i, j in zip(output_ub, self.preReluRange[y_name]['x_ub'])]
                self.preReluRange[y_name]['x_diff_lb'] = [max(i,j) for i, j in zip(out_diff_lb, self.preReluRange[y_name]['x_diff_lb'])]
                self.preReluRange[y_name]['x_diff_ub'] = [min(i,j) for i, j in zip(out_diff_ub, self.preReluRange[y_name]['x_diff_ub'])]
            else:
                self.preReluRange[y_name] = {}
                self.preReluRange[y_name]['lb'] = y_lb
                self.preReluRange[y_name]['ub'] = y_ub
                self.preReluRange[y_name]['diff_lb'] = y_diff_lb
                self.preReluRange[y_name]['diff_ub'] = y_diff_ub

                self.preReluRange[y_name]['x_lb'] = output_lb
                self.preReluRange[y_name]['x_ub'] = output_ub
                self.preReluRange[y_name]['x_diff_lb'] = out_diff_lb
                self.preReluRange[y_name]['x_diff_ub'] = out_diff_ub
            # self.preReluRange[y_name] = {}
            # self.preReluRange[y_name]['lb'] = y_lb
            # self.preReluRange[y_name]['ub'] = y_ub
            # self.preReluRange[y_name]['diff_lb'] = y_diff_lb
            # self.preReluRange[y_name]['diff_ub'] = y_diff_ub

        return output_lb, output_ub, out_diff_lb, out_diff_ub

    def range_propergation_3(self, input_lb, input_ub, diff_lb, diff_ub, window_size = 1, reset_bound = True):
        """
        input_lb, input_ub: the lower and upper bound of the input states, which forms the input space
        diff_lb, diff_ub: the lower and upper bound of the disturbance of each input state

        currently, only support one dimention input fully-connected NN

        return: the lower and upper bound of the global robustness f(x)-f(x')
        """

        if not self.lp_relax:
            print("For MILP exact range propergation, please call range_propergation_2() or range_propergation() (less efficient).")
            return None
        
        # reset the preReluRange
        self.preReluRange = {}

        input_lb = np.array(input_lb).flatten()
        input_ub = np.array(input_ub).flatten()
        diff_lb = np.array(diff_lb).flatten()
        diff_ub = np.array(diff_ub).flatten()

        lbs = []
        ubs = []
        dlbs = []
        dubs = []

        lbs.append(input_lb)
        ubs.append(input_ub)
        dlbs.append(diff_lb)
        dubs.append(diff_ub)

        self.time_0 = self.time_last_print = time.time()

        n_relu_layers = len(self.relu_ids) - 1
        for relu_layer_id in range(1, n_relu_layers + 1):
            in_layer = max(0, relu_layer_id - window_size)
            updateRange = False
            if relu_layer_id < n_relu_layers:
                updateRange = True
            input_lb, input_ub, diff_lb, diff_ub = self.range_prop_window(lbs[in_layer], ubs[in_layer], dlbs[in_layer], 
                                                                                   dubs[in_layer], relu_layer_id, 
                                                                                   window_size, updateRange=updateRange)
            if relu_layer_id + window_size <= n_relu_layers:
                lbs.append(input_lb)
                ubs.append(input_ub)
                dlbs.append(diff_lb)
                dubs.append(diff_ub)
        out_bound = [diff_lb, diff_ub]
        print("output bound (not difference bound)", [input_lb, input_ub])
        if self.lp_relax and window_size == 1:
            _min = 0
            _max = 0
            for y_name in self.preReluRange:
                ranges = [a+b for a,b in zip(self.preReluRange[y_name]['lb'], self.preReluRange[y_name]['diff_lb'])]
                _min = min(_min, min(ranges))
                ranges = [a+b for a,b in zip(self.preReluRange[y_name]['ub'], self.preReluRange[y_name]['diff_ub'])]
                _max = max(_max, max(ranges))
            print("range for big M", _min, _max)
        return out_bound
    
    def set_refine_config(self, window_size, percentage_diff,  max_diff, percentage_x = None, max_x = None):
        """
        window_size: the refine config is specified for each window size
        percentage_diff: percentage of diff relu nodes to be refined in each layer
        percentage_x: percentage of original relu nodes to be refined in each layer, None <=> percentage_x = percentage_diff
        max_diff: maximum number of diff relu nodes to be refined in each layer
        max_x: maximum number of original relu nodes to be refined in each layer, None  <=> max_x = max_diff
        """

        if percentage_x is None:
            percentage_x = percentage_diff
        if max_x is None:
            max_x = max_diff
        
        self.refine_config[window_size] = {}
        self.refine_config[window_size]['p_diff'] = percentage_diff
        self.refine_config[window_size]['p_orig'] = percentage_x
        self.refine_config[window_size]['max_diff'] = max_diff
        self.refine_config[window_size]['max_orig'] = max_x
        self.refine_config[window_size]['set_diff'] = set()
        self.refine_config[window_size]['set_orig'] = set()

    def get_refine_stat(self, window_size):
        if window_size not in self.refine_config:
            return [], []
        else:
            return list(self.refine_config[window_size]['set_orig']), list(self.refine_config[window_size]['set_diff'])
