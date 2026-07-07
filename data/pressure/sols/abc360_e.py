import sys

MOD = 998244353

def main():
    N, K = sys.stdin.read().split()
    N = int(N) % MOD
    K = int(K)
    invN = pow(N, MOD - 2, MOD)
    inv2 = pow(2, MOD - 2, MOD)
    a = (N - 2) * invN % MOD
    ak = pow(a, K, MOD)
    # p = a^k * (1 - 1/N) + 1/N
    p = (ak * ((1 - invN) % MOD) + invN) % MOD
    other = (N + 2) % MOD * inv2 % MOD
    ans = (p + (1 - p) % MOD * other) % MOD
    print(ans % MOD)

main()
