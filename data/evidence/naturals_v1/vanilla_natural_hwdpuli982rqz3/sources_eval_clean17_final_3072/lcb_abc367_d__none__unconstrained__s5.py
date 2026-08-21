import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input and split into a list of integers
    data = list(map(int, sys.stdin.read().split()))
    
    # Extract N, M and the sequence A
    N = data[0]
    M = data[1]
    A = data[2:]
    
    # Calculate prefix sums of A modulo M.
    # P[i] = (A_1 + ... + A_{i-1}) % M
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M
    P = list(accumulate([x % M for x in A], initial=0))
    
    # The distance clockwise from s to t is:
    # If s < t: (P[t-1] - P[s-1]) % M
    # If s > t: (P[N] - P[s-1] + P[t-1]) % M
    # In both cases, the distance is a multiple of M if:
    # P[t-1] ≡ P[s-1] (mod M)
    
    # We need to count pairs (s, t) such that 1 <= s, t <= N, s != t, and P[s-1] == P[t-1].
    # Note: P is indexed from 0 to N. The rest areas are 1 to N.
    # The prefix sums corresponding to rest areas 1...N are P[0]...P[N-1].
    
    # Count occurrences of each remainder in P[0...N-1]
    counts = Counter(P[:N])
    
    # For each remainder that appears 'c' times, we can form c * (c - 1) pairs
    # since s and t must be distinct.
    ans = sum(c * (c - 1) for c in counts.values())
    
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()