import sys
import numpy as np

data = sys.stdin.buffer.read().split(b'\n')
K = int(data[0])
S = data[1]
T = data[2]

a = np.frombuffer(S, dtype=np.uint8)
b = np.frombuffer(T, dtype=np.uint8)
ls = a.size
lt = b.size

ans = "No"
if ls == lt:
    if int((a != b).sum()) <= 1:
        ans = "Yes"
elif abs(ls - lt) == 1:
    # longer = A, shorter = B
    if ls > lt:
        A, B = a, b
    else:
        A, B = b, a
    lb = B.size
    mism = (A[:lb] != B)
    if not mism.any():
        ans = "Yes"
    else:
        p = int(mism.argmax())
        if bool((A[p+1:] == B[p:]).all()):
            ans = "Yes"

sys.stdout.write(ans + "\n")
