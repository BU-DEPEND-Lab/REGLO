import pickle
import matplotlib.pyplot as plt
import math

res = {}
with open('results0317_0.pickle', 'rb') as f:
    res = pickle.load(f)

ITNE_bounds = [[-1258.1572351579034, -1335.5525597059204, -1056.3562558863314, -914.0280802784973, -1207.0551721529355, -1137.8117524336176, -1286.3257228080074, -1240.4273921556578, -1325.77960674331, -1265.8127788128313], [1258.157235188668, 1335.5525597615197, 1056.3562558712529, 914.0280802835832, 1207.0551721130523, 1137.811752424972, 1286.3257227961294, 1240.4273921387557, 1325.7796066918381, 1265.812778846568]]

ts = res['t']
splits = res['split']
lbs = res['lb']
ubs = res['ub']
n_outputs = len(lbs[0])
n_res = len(lbs)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axs[0,0].set_ylabel("upper bounds")
axs[1,0].set_ylabel("lower bounds")
# axs[0,1].set_ylabel("upper bounds")
# axs[1,1].set_ylabel("lower bounds")
axs[0,0].set_xlabel("t (s)")
axs[1,0].set_xlabel("t (s)")
axs[0,1].set_xlabel("# of splits")
axs[1,1].set_xlabel("# of splits")
for i in range(n_outputs):
    l = [lbs[j][i] for j in range(n_res)]
    u = [ubs[j][i] for j in range(n_res)]
    axs[0,0].plot(ts, u, f"C{i}", label = f"output_{i}")
    axs[0,1].plot(splits, u, f"C{i}", label = f"output_{i}")
    axs[1,0].plot(ts, l, f"C{i}", label = f"output_{i}")
    axs[1,1].plot(splits, l, f"C{i}", label = f"output_{i}")
    diff_itne = ITNE_bounds[1][i] - ITNE_bounds[0][i]
    diff = u[-1] - l[-1]
    print(f"[{math.floor(ITNE_bounds[0][i])}, {math.ceil(ITNE_bounds[1][i])}],\t [{math.floor(l[-1])}, {math.ceil(u[-1])}], {100*(diff_itne-diff)/diff_itne:.1f}%")
axs[0,0].legend(ncol=2)
axs[0,1].legend(ncol=2)
axs[1,0].legend(ncol=2)
axs[1,1].legend(ncol=2)
plt.savefig('results0317_0.png')

print(f"lbs = {lbs[-1]}, ubs = {ubs[-1]}")