import sys
from functools import reduce

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    MOD = 998244353

    # We need to compute sum_{k=0}^N popcount(k & M)
    # This is equivalent to sum_{i=0}^{59} [i-th bit of M is 1] * (count of k in [0, N] where i-th bit is 1)
    
    # For a fixed bit i, the number of integers k in [0, N] that have the i-th bit set:
    # The bits follow a pattern: 2^i zeros, then 2^i ones.
    # Full blocks of size 2^(i+1) contain 2^i ones.
    # The remaining part is N + 1 modulo 2^(i+1).
    # If the remainder is > 2^i, the number of ones in the remainder is (remainder - 2^i).
    
    # We use a list comprehension to calculate the contribution of each bit i from 0 to 59.
    # Then we use reduce to sum them up modulo 998244353.
    
    # Logic for count of k in [0, N] with i-th bit set:
    # full_blocks = (N + 1) // (2**(i+1))
    # remainder = (N + 1) % (2**(i+1))
    # count = full_blocks * (2**i) + max(0, remainder - 2**i)
    
    # We only care about bits i where (M >> i) & 1 is true.
    
    ans = reduce(
        lambda acc, i: (acc + (
            ( ((N + 1) // (2**(i+1))) * (2**i) + max(0, ((N + 1) % (2**(i+1))) - 2**i) )
            if (M >> i) & 1 else 0
        )) % MOD,
        range(60),
        0
    )
    
    print(ans)

if __name__ == "__main__":
    solve()