import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Prefix sums of distances: P[i] is distance from area 1 to area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN]
    # We use accumulate to avoid explicit for-loops.
    # We append 0 at the start to represent the starting point of area 1.
    P = list(accumulate([0] + A))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    # Both conditions are equivalent to: P[t-1] % M == P[s-1] % M
    # provided we consider the indices carefully.
    
    # Let R be the list of prefix sums modulo M for areas 1 to N.
    # Area i corresponds to P[i-1].
    R = [p % M for p in P[:N]]
    
    # Count occurrences of each remainder
    counts = Counter(R)
    
    # For each remainder r, if there are k areas with that remainder,
    # there are k * (k - 1) pairs (s, t) such that the distance is a multiple of M.
    # We use a generator expression inside sum() to avoid explicit loops.
    ans = sum(k * (k - 1) for k in counts.values())
    
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()