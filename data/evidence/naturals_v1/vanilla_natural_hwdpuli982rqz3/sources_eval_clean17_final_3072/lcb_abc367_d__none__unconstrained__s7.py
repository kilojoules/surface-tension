import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    # Parse A_i values
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances modulo M
    # P[i] = (A_1 + ... + A_{i-1}) % M
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M
    # Note: accumulate returns an iterator, we convert to list
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    
    # The distance clockwise from s to t is:
    # If s < t: (P[t-1] - P[s-1]) % M
    # If s > t: (P[N] - P[s-1] + P[t-1]) % M
    # We want this distance to be 0 mod M.
    # In both cases, the condition is: P[t-1] ≡ P[s-1] (mod M)
    # However, we must exclude the case where s = t.
    
    # We only care about P[0] through P[N-1] because P[N] is the total loop.
    # Let's take the first N prefix sums.
    prefix_sums = P[:N]
    
    # Count occurrences of each remainder modulo M
    counts = Counter(prefix_sums)
    
    # For each remainder r, if it appears C times, we can pick 2 distinct 
    # indices (s-1, t-1) such that P[s-1] == P[t-1] == r.
    # One will be s, the other t. Since s != t, we have C * (C - 1) pairs.
    # Note: The problem defines clockwise distance. If P[s-1] == P[t-1],
    # then (P[t-1] - P[s-1]) % M == 0.
    # If s < t, distance is P[t-1] - P[s-1].
    # If s > t, distance is (P[N] - P[s-1]) + P[t-1].
    # For this to be 0 mod M, we need P[N] to be 0 mod M if we use the logic P[s-1] == P[t-1].
    # Actually, the condition is:
    # Distance(s, t) = (P[t-1] - P[s-1]) mod (TotalSum)
    # But we only need Distance(s, t) % M == 0.
    # Distance(s, t) % M = (P[t-1] - P[s-1]) % M.
    # This is 0 if and only if P[t-1] ≡ P[s-1] (mod M).
    
    # The number of pairs (s, t) with s != t such that P[s-1] ≡ P[t-1] (mod M) is:
    # Sum of (count * (count - 1)) for all unique remainders.
    
    ans = sum(c * (c - 1) for c in counts.values())
    
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()