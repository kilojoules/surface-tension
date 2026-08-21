import sys

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    m = int(input_data[1])
    MOD = 998244353

    # The goal is to compute sum_{k=0}^N popcount(k & M)
    # This is equivalent to sum_{i=0}^{59} [i-th bit of M is 1] * (count of k in [0, N] where i-th bit is 1)
    
    # For a fixed bit i, the i-th bit of k is 1 if (k // 2^i) is odd.
    # In the range [0, N], the i-th bit follows a pattern: 2^i zeros, then 2^i ones.
    # The number of 1s in the i-th bit for k in [0, N] is:
    # full_blocks * 2^i + max(0, (N + 1) % (2^(i+1)) - 2^i)
    # where full_blocks = (N + 1) // (2^(i+1))
    
    # We use a list comprehension to calculate the contribution of each bit i from 0 to 60
    # and sum them up.
    
    ans = sum([
        (
            # If the i-th bit of M is 0, it contributes nothing
            0 if not (m & (1 << i)) else (
                # Calculate count of k in [0, N] with i-th bit set
                (
                    ((n + 1) // (1 << (i + 1))) * (1 << i) + 
                    max(0, ((n + 1) % (1 << (i + 1))) - (1 << i))
                ) % MOD
            )
        )
        for i in range(62)
    ]) % MOD

    print(ans)

if __name__ == "__main__":
    solve()