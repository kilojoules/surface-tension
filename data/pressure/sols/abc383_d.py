import sys, math
import numpy as np

def main():
    N = int(sys.stdin.readline())
    M = math.isqrt(N)          # need primes p,q with p*q <= sqrt(N)
    L = max(M, 2)              # sieve limit

    # ---- vectorized sieve of Eratosthenes (no python loops) ----
    A = math.isqrt(L)
    a = np.arange(2, A + 1, dtype=np.int64)
    if a.size:
        counts = (L // a) - 1                     # k = 2 .. L//a  (>=1 each)
        offs_a = np.repeat(a, counts)
        cum = np.cumsum(counts)
        group_start = np.repeat(cum - counts, counts)
        within = np.arange(offs_a.size, dtype=np.int64) - group_start
        composites = offs_a * (within + 2)        # all <= L
    else:
        composites = np.empty(0, dtype=np.int64)

    is_prime = np.ones(L + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    is_prime[composites] = False
    primes = np.nonzero(is_prime)[0]

    # ---- numbers with exactly 9 divisors ----
    # form 1: p^8  (divisors = 9)
    small = primes[primes < 100]
    c8 = int(np.count_nonzero(small.astype(object) ** 8 <= N))

    # form 2: p^2 * q^2 = (p*q)^2 with p<q primes, i.e. p*q <= M
    upper = M // np.maximum(primes, 1)            # max q value for each p
    hi = np.searchsorted(primes, upper, side='right')   # #primes <= upper
    lo = np.searchsorted(primes, primes, side='right')  # #primes <= p
    cnt = hi - lo                                 # primes q with p < q <= upper
    cnt = np.where(cnt > 0, cnt, 0)
    pairs = int(cnt.sum())

    print(c8 + pairs)

main()
