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
    # The number of k in [0, N] with the i-th bit set can be calculated as:
    # Full blocks of size 2^(i+1) contain 2^i ones.
    # The remaining part is max(0, (N % 2^(i+1)) - 2^i + 1)
    
    # We use a list comprehension to iterate through all possible bit positions (0 to 60)
    # and sum the contributions of bits that are set in M.
    
    ans = sum([
        (
            # Count of k in [0, N] that have the i-th bit set
            ((n + 1) // (2**(i + 1)) * (2**i)) + 
            max(0, (n + 1) % (2**(i + 1)) - (2**i))
        )
        for i in range(61) 
        if (m >> i) & 1
    ])
    
    # Output the result modulo 998244353
    print(ans % MOD)

if __name__ == "__main__":
    solve()