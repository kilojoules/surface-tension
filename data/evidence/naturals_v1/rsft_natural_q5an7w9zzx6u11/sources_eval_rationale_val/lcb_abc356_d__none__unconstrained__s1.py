import sys

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    m = int(input_data[1])
    mod = 998244353

    # For a bit position i, the i-th bit is 1 every 2^i integers 
    # after an initial block of 2^i zeros.
    # The pattern repeats every 2^(i+1).
    # Number of set bits in range [0, N] for bit i:
    # full_blocks = (N + 1) // 2^(i+1)
    # remaining = (N + 1) % 2^(i+1)
    # count = full_blocks * 2^i + max(0, remaining - 2^i)
    
    # We only care about bit i if the i-th bit of M is 1.
    # We use a generator expression inside sum() to avoid explicit loops.
    # We iterate through bit positions 0 to 60.
    
    ans = sum(
        (
            (( (n + 1) // (1 << (i + 1)) ) * (1 << i)) + 
            max(0, ((n + 1) % (1 << (i + 1))) - (1 << i))
        ) 
        for i in range(61) 
        if (m >> i) & 1
    ) % mod

    print(ans)

if __name__ == "__main__":
    solve()