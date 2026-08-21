import sys

def solve():
    # Read N and M from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    m = int(input_data[1])
    mod = 998244353

    # The total sum is the sum of contributions of each bit i that is set in M.
    # For a bit i, we count how many k in [0, N] have the i-th bit set.
    # The number of integers k in [0, N] is N + 1.
    # In every block of 2^(i+1) numbers, 2^i numbers have the i-th bit set.
    # The remaining numbers are (N + 1) % 2^(i+1).
    # If the remainder is greater than 2^i, the additional numbers with the i-th bit set
    # are (N + 1) % 2^(i+1) - 2^i.
    
    # We use a list comprehension to iterate through bits 0 to 60.
    # We filter for bits that are set in M.
    
    ans = sum([
        (
            ((n + 1) // (2**(i + 1))) * (2**i) + 
            max(0, ((n + 1) % (2**(i + 1))) - (2**i))
        )
        for i in range(61) if (m >> i) & 1
    ])
    
    # Output the result modulo 998244353
    print(ans % mod)

if __name__ == "__main__":
    solve()