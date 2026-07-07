import sys

p = 998244353
N = int(sys.stdin.readline())
d = len(str(N))
B = pow(10, d, p)
num = (pow(B, N, p) - 1) % p
den = (B - 1) % p
if den == 0:
    S = N % p
else:
    S = num * pow(den, p - 2, p) % p
ans = N % p * S % p
print(ans)
