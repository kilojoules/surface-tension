import sys
import numpy as np
import mpmath as mp

mp.mp.prec = 64
HALF = mp.mpf(1) / 2
one = mp.mpf(1)

data = sys.stdin.read().split()
N = int(data[0]); S = int(data[1]); T = int(data[2])
coords = np.array(data[3:3 + 4 * N], dtype=np.int64).reshape(N, 4)
Px = coords[:, 0]; Py = coords[:, 1]
Qx = coords[:, 2]; Qy = coords[:, 3]

# permutations via meshgrid filter
grids = np.meshgrid(*([np.arange(N)] * N), indexing='ij')
cand = np.stack(grids, axis=-1).reshape(-1, N)
valid = (np.sort(cand, axis=1) == np.arange(N)).all(axis=1)
perms = cand[valid]                                      # (nP, N)

bits = (np.arange(2 ** N)[:, None] >> np.arange(N)) & 1  # (nO, N)
entryX = np.where(bits == 0, Px[None, :], Qx[None, :])
entryY = np.where(bits == 0, Py[None, :], Qy[None, :])
exitX = np.where(bits == 0, Qx[None, :], Px[None, :])
exitY = np.where(bits == 0, Qy[None, :], Py[None, :])

eX = entryX[:, perms]; eY = entryY[:, perms]             # (nO, nP, N)
xX = exitX[:, perms]; xY = exitY[:, perms]

# move squared distances in path order: from origin to first entry, then exit_k -> entry_{k+1}
prevX = np.concatenate([np.zeros_like(xX[..., :1]), xX[..., :-1]], axis=-1)
prevY = np.concatenate([np.zeros_like(xY[..., :1]), xY[..., :-1]], axis=-1)
move_sq = (eX - prevX) ** 2 + (eY - prevY) ** 2          # (nO, nP, N)

len_sq = ((Px - Qx) ** 2 + (Py - Qy) ** 2)[perms]        # (nP, N)
len_sq = np.broadcast_to(len_sq, eX.shape)               # (nO, nP, N)

move_d = ((move_sq.astype(object) * one) ** HALF) / S
draw_d = ((len_sq.astype(object) * one) ** HALF) / T

inter = np.stack([move_d, draw_d], axis=-1).reshape(move_d.shape[:-1] + (2 * N,))
totals = inter.sum(axis=-1)                              # (nO, nP)
ans = totals.min()

mp.mp.prec = 260
scaled = ans * mp.power(10, 20)
k = int(mp.nint(scaled))
s = str(k)
if len(s) <= 20:
    s = '0' * (21 - len(s)) + s
print(s[:-20] + '.' + s[-20:])
