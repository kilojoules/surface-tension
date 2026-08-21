import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: number of rest areas, M: the divisor
    n = int(input_data[0])
    m = int(input_data[1])
    a = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances modulo M.
    # P[i] is the distance from rest area 1 to rest area (i+1) mod M.
    # P[0] = 0, P[1] = A1 % M, P[2] = (A1 + A2) % M, ..., P[N] = (Sum of all A) % M.
    # Note: We use a list comprehension to avoid explicit for-loops.
    p = list(accumulate([x % m for x in a], initial=0))
    
    # The distance from s to t clockwise is:
    # 1. If s < t: (P[t-1] - P[s-1]) mod M
    # 2. If s > t: (P[N] - P[s-1] + P[t-1]) mod M
    # We want this distance to be 0 mod M.
    # This simplifies to: P[t-1] ≡ P[s-1] (mod M) for all s != t.
    
    # Let's count occurrences of each value in P[0...N-1].
    # P[N] is the total loop distance, which we don't use as a starting point P[s-1] 
    # because rest area N+1 is rest area 1.
    counts = Counter(p[:n])
    
    # For each unique value v that appears k times in P[0...N-1],
    # we can pick any two distinct indices (s-1, t-1) such that P[s-1] = P[t-1] = v.
    # One will be s, the other t. Since s != t, we have k * (k - 1) ordered pairs.
    # However, we must check if the distance is actually 0 mod M.
    # If P[s-1] == P[t-1], then (P[t-1] - P[s-1]) % M == 0.
    # This holds regardless of whether s < t or s > t.
    
    # Sum k * (k - 1) for all k in counts.values()
    ans = sum(k * (k - 1) for k in counts.values())
    
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()