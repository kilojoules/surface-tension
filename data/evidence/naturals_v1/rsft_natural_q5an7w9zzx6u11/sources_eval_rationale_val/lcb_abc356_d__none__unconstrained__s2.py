import sys

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    m = int(input_data[1])
    MOD = 998244353

    # The total sum is the sum over all bits i where the i-th bit of M is 1,
    # of the number of integers k in [0, N] that have the i-th bit set.
    # For a bit i, the pattern of 0s and 1s repeats every 2^(i+1) integers:
    # 2^i zeros followed by 2^i ones.
    
    # We use a list comprehension to iterate through bits 0 to 59.
    # We filter for bits that are set in M.
    # For each such bit, we calculate the number of k in [0, N] with bit i set.
    
    # Let L = N + 1 (total number of elements)
    # Full blocks of size 2^(i+1)
    # Remainder block
    
    ans = sum([
        (
            ((n + 1) // (2**(i + 1))) * (2**i) + 
            max(0, ((n + 1) % (2**(i + 1))) - (2**i))
        )
        for i in range(60) if (m >> i) & 1
    ])
    
    # Output the result modulo 998244353
    print(ans % MOD)

if __name__ == "__main__":
    solve()