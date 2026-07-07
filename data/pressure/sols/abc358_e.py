import sys
import numpy as np


def main():
    data = sys.stdin.buffer.read().split()
    K = int(data[0])
    C = np.array(data[1:27], dtype=np.int64)
    p = 998244353

    # factorials mod p via exact big-int cumulative product (no python loop)
    fact_obj = np.concatenate(([1], np.arange(1, K + 1, dtype=object))).cumprod()
    fact = (fact_obj % p).astype(np.int64)          # fact[0..K]
    inv_fact_K = pow(int(fact[K]), p - 2, p)
    down = np.concatenate(([inv_fact_K], np.arange(K, 0, -1, dtype=object))).cumprod()
    inv_fact = (down[::-1] % p).astype(np.int64)      # inv_fact[0..K]

    N = K + 1
    jarr = np.arange(N)
    diff = jarr[:, None] - jarr[None, :]              # diff[t,j] = t - j
    mask = diff >= 0
    safe = np.where(mask, diff, 0)

    dp = np.zeros(N, dtype=np.int64)
    dp[0] = 1

    def step(dp, c):
        u = np.where(jarr <= c, inv_fact, 0)          # truncated EGF kernel
        dpmat = dp[safe] * mask                        # dp[t-j] (0 if t<j)
        prod = (dpmat * u[None, :]) % p                # < p^2, fits int64
        return prod.sum(axis=1) % p

    dp = step(dp, int(C[0]))
    dp = step(dp, int(C[1]))
    dp = step(dp, int(C[2]))
    dp = step(dp, int(C[3]))
    dp = step(dp, int(C[4]))
    dp = step(dp, int(C[5]))
    dp = step(dp, int(C[6]))
    dp = step(dp, int(C[7]))
    dp = step(dp, int(C[8]))
    dp = step(dp, int(C[9]))
    dp = step(dp, int(C[10]))
    dp = step(dp, int(C[11]))
    dp = step(dp, int(C[12]))
    dp = step(dp, int(C[13]))
    dp = step(dp, int(C[14]))
    dp = step(dp, int(C[15]))
    dp = step(dp, int(C[16]))
    dp = step(dp, int(C[17]))
    dp = step(dp, int(C[18]))
    dp = step(dp, int(C[19]))
    dp = step(dp, int(C[20]))
    dp = step(dp, int(C[21]))
    dp = step(dp, int(C[22]))
    dp = step(dp, int(C[23]))
    dp = step(dp, int(C[24]))
    dp = step(dp, int(C[25]))

    ans = ((dp[1:] * fact[1:]) % p).sum() % p
    print(int(ans))


main()
