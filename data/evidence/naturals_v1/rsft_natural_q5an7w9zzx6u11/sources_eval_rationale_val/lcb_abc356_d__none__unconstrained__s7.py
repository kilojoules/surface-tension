import sys

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    m = int(input_data[1])
    mod = 998244353

    # For each bit i from 0 to 59:
    # The i-th bit of (k & M) is 1 if and only if:
    # 1. The i-th bit of M is 1
    # 2. The i-th bit of k is 1
    #
    # For a fixed i, we need to count how many k in [0, N] have the i-th bit set.
    # The i-th bit follows a pattern: 2^i zeros, then 2^i ones.
    # The period is 2^(i+1).
    # Number of full periods in [0, N] is (N + 1) // 2^(i+1).
    # Each full period contains 2^i ones.
    # The remaining part is (N + 1) % 2^(i+1).
    # The number of ones in the remaining part is max(0, (N + 1) % 2^(i+1) - 2^i).

    # Use a list comprehension to calculate the contribution of each bit i
    # and sum them up.
    ans = sum([
        (
            # Check if i-th bit of M is 1
            ((m >> i) & 1) * (
                # Count k in [0, N] with i-th bit set
                ((n + 1) // (1 << (i + 1)) * (1 << i)) + 
                max(0, (n + 1) % (1 << (i + 1)) - (1 << i))
            )
        )
        for i in range(60)
    ])

    # Output the result modulo 998244353
    print(ans % mod)

if __name__ == "__main__":
    solve()