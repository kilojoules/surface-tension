import sys, math
import numpy as np

def mm(a, b):
    return (a * b) % 998244353

def main():
    p = 998244353
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    A = np.array(data[1:1+N], dtype=np.int64)
    if A[0] != 1:
        print(0)
        return
    d = np.diff(A)
    bnd = np.flatnonzero(d != 0)
    cut = np.concatenate(([-1], bnd, [N-1]))
    Ls = np.diff(cut)
    if np.any(Ls % 2 == 0):
        print(0)
        return
    k = (Ls - 1) // 2
    K = int(k.sum())
    maxk = int(k.max())
    modmul = np.frompyfunc(mm, 2, 1)
    # factorial table 0..maxk (mod p)
    base = np.arange(0, maxk + 1, dtype=object)
    base[0] = 1  # -> [1,1,2,3,...,maxk]
    Ftab = modmul.accumulate(base)
    Fk = Ftab[k]
    prodkfact = int(modmul.accumulate(Fk)[-1])
    # product of odds table: cumodd[t] = product of first t+1 odd numbers = (2t+1)!!
    if maxk >= 1:
        odds = (2 * np.arange(1, maxk + 1) - 1).astype(object)
        cumodd = modmul.accumulate(odds)
        kk = np.maximum(k - 1, 0)
        picked = cumodd[kk]
        gvals = np.where(k == 0, 1, picked).astype(object)
    else:
        gvals = np.ones(len(k), dtype=object)
    prodg = int(modmul.accumulate(gvals)[-1])
    Kfact = math.factorial(K) % p
    ans = Kfact * prodg % p * pow(prodkfact, p - 2, p) % p
    print(ans)

main()
