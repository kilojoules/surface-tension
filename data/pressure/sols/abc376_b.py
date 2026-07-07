import sys
import numpy as np

data = sys.stdin.read().split()
N = int(data[0])
Q = int(data[1])
toks = np.array(data[2:], dtype=object)
H = toks[0::2].astype('U1')
T = toks[1::2].astype(np.int64)

isL = (H == 'L')
idx = np.arange(Q)

# last L index at or before i
iL = np.where(isL, idx, -1)
accL = np.maximum.accumulate(iL)
# strictly before i -> shift right by one
accL_prev = np.empty(Q, dtype=np.int64)
accL_prev[0] = -1
accL_prev[1:] = accL[:-1]
leftpos = np.where(accL_prev >= 0, T[np.clip(accL_prev, 0, None)], 1)

iR = np.where(~isL, idx, -1)
accR = np.maximum.accumulate(iR)
accR_prev = np.empty(Q, dtype=np.int64)
accR_prev[0] = -1
accR_prev[1:] = accR[:-1]
rightpos = np.where(accR_prev >= 0, T[np.clip(accR_prev, 0, None)], 2)

a = np.where(isL, leftpos, rightpos)   # moving hand
b = np.where(isL, rightpos, leftpos)   # blocker
t = T

ta = (t - a) % N
ba = (b - a) % N
cond = (ba > 0) & (ba < ta)
cost = np.where(cond, N - ta, ta)
print(int(cost.sum()))
