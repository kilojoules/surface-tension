import sys, numpy as np
data = sys.stdin.buffer.read().split()
N = int(data[0]); M = int(data[1])
X = np.array(data[2:2+M], dtype=np.int64)
A = np.array(data[2+M:2+2*M], dtype=np.int64)
if int(A.sum()) != N:
    print(-1)
else:
    order = np.argsort(X)
    Xs = X[order]; As = A[order]
    pref_excl = np.cumsum(As) - As
    if bool(np.all(Xs <= pref_excl + 1)):
        total = (N * (N + 1)) // 2 - int((A * X).sum())
        print(total)
    else:
        print(-1)
