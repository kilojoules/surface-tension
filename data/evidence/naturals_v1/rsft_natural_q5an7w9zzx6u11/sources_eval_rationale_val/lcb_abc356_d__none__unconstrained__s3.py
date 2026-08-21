import sys

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    m = int(input_data[1])
    mod = 998244353

    # For a bit i, it contributes to the sum if the i-th bit of M is 1.
    # If the i-th bit of M is 1, we need to count how many k in [0, N] 
    # have the i-th bit set.
    # The i-th bit (value 2^i) follows a pattern: 2^i zeros, then 2^i ones.
    # The period is 2^(i+1).
    # Number of full periods in [0, N] is (N + 1) // 2^(i+1).
    # Each full period contains 2^i ones.
    # The remaining part is (N + 1) % 2^(i+1).
    # The number of ones in the remaining part is max(0, (N + 1) % 2^(i+1) - 2^i).
    
    # We use a list comprehension to calculate the contribution of each bit from 0 to 59.
    # We only consider bits where (m >> i) & 1 is true.
    
    ans = sum([
        (
            ((n + 1) // (1 << (i + 1)) * (1 << i)) + 
            max(0, ((n + 1) % (1 << (i + 1))) - (1 << i))
        )
        for i in range(60) 
        if (m >> i) & 1
    ])
    
    print(ans % mod)

if __name__ == "__main__":
    solve()