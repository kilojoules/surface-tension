import sys
import numpy as np

def main():
    data=sys.stdin.buffer.read().split()
    N=int(data[0]); K=int(data[1])
    if N==1:
        out=np.ones(K,dtype=np.int64)
    elif N%2==0:
        m=N//2
        # blocks: [m]x1 ; N..m+1 xK ; [m]x(K-1) ; m-1..1 xK
        valsB=np.arange(N,m,-1,dtype=np.int64)          # N..m+1
        valsD=np.arange(m-1,0,-1,dtype=np.int64)         # m-1..1
        vals=np.concatenate(([m],valsB,[m],valsD)).astype(np.int64)
        reps=np.concatenate(([1],np.full(valsB.size,K),[K-1],np.full(valsD.size,K))).astype(np.int64)
        out=np.repeat(vals,reps)
    else:
        m=(N+1)//2
        # blocks: [m]xK ; [m-1]x1 ; N..m+1 xK ; [m-1]x(K-1) ; m-2..1 xK
        valsB=np.arange(N,m,-1,dtype=np.int64)           # N..m+1
        valsD=np.arange(m-2,0,-1,dtype=np.int64)          # m-2..1
        vals=np.concatenate(([m],[m-1],valsB,[m-1],valsD)).astype(np.int64)
        reps=np.concatenate(([K],[1],np.full(valsB.size,K),[K-1],np.full(valsD.size,K))).astype(np.int64)
        out=np.repeat(vals,reps)
    sys.stdout.write(' '.join(out.astype(str)))
    sys.stdout.write('\n')

main()
