from NeurNet import NeurNet, load_model, load_extracted_model
from SplitConfig import SplitConfig
import heapq as hq
from collections import deque
import math
import numpy as np
import time
from typing import List

class BnB:
    def __init__(self, model_name = 'test', batch_size = 1, dbg_id = None):
        self.batch_size = batch_size
        self.NN_worker = NeurNet(model_name, batch_size=batch_size, alpha=True, beta=True, dbg_id=dbg_id)
        self.out_size = 0

    def setupNeuralNetwork(self, NN_layers, input_lb, input_ub, dist_lb, dist_ub):
        self.NN_worker.setupNeuralNetwork(NN_layers, input_lb, input_ub, dist_lb, dist_ub)
        self.out_size = self.NN_worker.last_layer_size()
    
    def setupSplitConfig(self, configs: List[SplitConfig]):
        self.NN_worker.reset_beta_config()
        for i in range(len(configs)):
            configs[i].setConfig(self.NN_worker, batch_id=i)

    def branchnBound(self, n_splits = 1000, timeout = 3600):

        out_lb_pqs = [] # the i-th priority queue entry: (dlb_i, split_idx)
        out_ub_pqs = [] # the i-th priority queue entry: (-dub_i, split_idx)
        for _ in range(self.out_size):
            out_lb_pqs.append([])
            out_ub_pqs.append([])
        undecide = deque()
        split_idx = 0
        last_recorded_split_id = -1
        split_config_map = {}
        split_lb_map = {}       # split -1 is a virtual parent split
        split_ub_map = {}
        removed_split = set()
        leaf_split = set() # splits that cannot split anymore
        evaled_split = set() # split config hashes for evaulated splits

        def _getSplitableTop(pq):
            while pq:
                _, idx = pq[0]
                if idx in leaf_split:
                    return None
                elif idx in removed_split:
                    hq.heappop(pq)
                else:
                    return idx
            return None

        def _nextSplitCandidate():
            candidates = []
            for pq in out_lb_pqs + out_ub_pqs:
                idx = _getSplitableTop(pq)
                if idx is not None:
                    candidates.append(idx)
            if candidates:
                return np.random.choice(candidates)
            else:
                return None

        def _getBound(pq):
            while pq:
                bound, idx = pq[0]
                if idx in removed_split:
                    hq.heappop(pq)
                else:
                    return bound
            return -math.inf
        def _outDistBounds():
            lbs = []
            for pq in out_lb_pqs:
                lbs.append(_getBound(pq))
            ubs = []
            for pq in out_ub_pqs:
                ubs.append(-_getBound(pq))
            return lbs, ubs

        lb_loss = lambda p_split: None if p_split is None else -np.sum(split_lb_map[p_split])
        ub_loss = lambda p_split: None if p_split is None else np.sum(split_ub_map[p_split])

        undecide.append((SplitConfig(), None))
        terminate = False
        t = time.time()
        rounds = 0
        need_tmp_result = True
        results = {'split': [], 't': [], 'lb': [], 'ub': []}
        while len(undecide) >  0 or (split_idx < n_splits and not terminate): 
            configs = []
            parent_splits = []
            split_idxs = []
            while len(configs) < self.batch_size:
                while len(undecide) > 0 and len(configs) < self.batch_size:
                    config, parent_split = undecide.popleft()
                    split_config_map[split_idx] = config
                    if config.hash in evaled_split:
                        # print("DBG: congrat! split config already evaluated.")
                        continue
                    evaled_split.add(config.hash)
                    configs.append(config)
                    parent_splits.append(parent_split)
                    split_idxs.append(split_idx)
                    split_idx += 1
                if len(configs) < self.batch_size and not terminate and not need_tmp_result:
                    next_split = _nextSplitCandidate()
                    if next_split is not None:
                        removed_split.add(next_split)
                        # print("Split", next_split)
                        config = split_config_map[next_split]
                        for new_config in config.splitNewNode():
                            undecide.append((new_config, next_split))
                    else:
                        break
                else:
                    break
            # while len(undecide) > 0:
            #     config, parent_split = undecide.popleft()
            #     split_config_map[split_idx] = config
            #     if config.hash in evaled_split:
            #         print("DBG: congrat! split config already evaluated.")
            #         continue
            #     evaled_split.add(config.hash)
            self.setupSplitConfig(configs)
            dlbs, dubs = self.NN_worker.narrow_the_dist_bound()
            candidates, _ = self.NN_worker.split_candidate()
            for i in range(len(configs)):
                config = configs[i]
                parent_split = parent_splits[i]
                dlb = dlbs[i]
                dub = dubs[i]
                split_i = split_idxs[i]
                candidate = candidates[i]
                if parent_split is None:
                    new_dlb = dlb.numpy()
                    new_dub = dub.numpy()
                else:
                    new_dlb = np.maximum(split_lb_map[parent_split], dlb.numpy())
                    new_dub = np.minimum(split_ub_map[parent_split], dub.numpy())
                split_lb_map[split_i] = new_dlb
                split_ub_map[split_i] = new_dub
                for i in range(self.out_size):
                    hq.heappush(out_lb_pqs[i], (new_dlb[i], split_i))
                    hq.heappush(out_ub_pqs[i], (-new_dub[i], split_i))
                # print(f'[split {split_i}] output variation bound:', np.min(new_dlb), np.max(new_dub))
                if candidate is not None:
                    config.set_next_candidate(candidate)
                else:
                    leaf_split.add(split_i)
            if rounds > 10:
                need_tmp_result = True
            else:
                rounds += 1
            if need_tmp_result and len(undecide) == 0:
                if last_recorded_split_id == split_idx:
                    print("BnB finished successfully. Found exact bounds.")
                    terminate = True
                    break
                lbs, ubs = _outDistBounds()
                results['split'].append(split_idx)
                results['t'].append(time.time() - t)
                results['lb'].append(lbs)
                results['ub'].append(ubs)
                print(f'[split {split_idx}] current global output variation bound: \nlbs = {lbs}, \nubs = {ubs}', flush=True)
                need_tmp_result = False
                rounds = 0
            if time.time() - t > timeout:
                terminate = True
        lbs, ubs = _outDistBounds()
        results['split'].append(split_idx)
        results['t'].append(time.time() - t)
        results['lb'].append(lbs)
        results['ub'].append(ubs)
        print(f"After {time.time() - t} s, final output variation bounds becomes \nlbs = {lbs}, \nubs = {ubs}.", flush=True)
        return results


def main():
    # model_name = '4c2d_noBN_4c2d_reg_1e_3_0'
    # input_lb = imageio.imread("data/cifar_lb.png") / 255.0
    # input_ub = imageio.imread("data/cifar_ub.png") / 255.0
    # print(np.max(input_ub), np.min(input_lb))

    # eps = 1.0/255.0
    eps = 1e-3
    # input_lb = np.clip(input_lb - eps, 0.0, 1.0)
    # input_ub = np.clip(input_ub + eps, 0.0, 1.0)
    input_lb = np.zeros((28,28,1))
    input_ub = np.ones((28,28,1))
    diff_lb = -eps * np.ones_like(input_lb)
    diff_ub = eps * np.ones_like(input_lb)
    # Layers = load_extracted_model(model_name)
    Layers, model_name = load_model()

    test_bnb = BnB(model_name)
    test_bnb.setupNeuralNetwork(Layers, diff_lb, diff_ub)
    res_lbs, res_ubs = test_bnb.branchnBound()
    # n_dbg = 4*5*5
    # lines = ["" for _ in range(n_dbg)]
    # for i in np.random.permutation(n_dbg):
    #     test_bnb = BnB(model_name,dbg_id=i)
    #     test_bnb.setupNeuralNetwork(Layers, diff_lb, diff_ub)
    #     res_lbs, res_ubs = test_bnb.branchnBound()
    #     lines[i] = f'{i}:\t{res_lbs[0][0]:.4f}\t{res_ubs[0][0]:.4f}\t{res_lbs[1][0]:.4f}\t{res_ubs[1][0]:.4f}\t{res_lbs[2][0]:.4f}\t{res_ubs[2][0]:.4f}'
    #     with open('dbg_results_4.log', 'w') as f:
    #         f.writelines('\n'.join(lines))

if __name__ == '__main__':
    main()
