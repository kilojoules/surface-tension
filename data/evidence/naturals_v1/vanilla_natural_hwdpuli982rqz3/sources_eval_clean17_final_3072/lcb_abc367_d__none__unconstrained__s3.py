import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map input to integers
    # N: number of rest areas, M: the divisor
    # A: list of steps between rest areas
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of A modulo M.
    # P[i] = (A_1 + ... + A_{i-1}) % M
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M
    # The distance clockwise from s to t is:
    # 1. If s < t: (P[t-1] - P[s-1]) % M
    # 2. If s > t: (P[N] - P[s-1] + P[t-1]) % M
    
    P = list(accumulate(A, initial=0))
    # We only care about values modulo M
    P_mod = [x % M for x in P]
    
    # Total distance around the lake
    S = P_mod[N]
    
    # We want to find pairs (s, t) such that distance % M == 0.
    # Let x = P_mod[s-1] and y = P_mod[t-1].
    # If s < t: (y - x) % M == 0  => y == x
    # If s > t: (S - x + y) % M == 0 => y == (x - S) % M
    
    # Count occurrences of each remainder in P_mod[0...N-1]
    # Note: P_mod has N+1 elements, we only need the first N for the starting/ending points.
    counts = Counter(P_mod[:N])
    
    # For each unique remainder 'x', we have 'count' positions.
    # 1. Pairs (s, t) where s < t and P_mod[s-1] == P_mod[t-1]:
    #    This is combination(count, 2) = count * (count - 1) // 2
    # 2. Pairs (s, t) where s > t and P_mod[t-1] == (P_mod[s-1] - S) % M:
    #    Let y = (x - S) % M. The number of pairs is count(x) * count(y).
    #    Special case: if x == y (which happens if S % M == 0), 
    #    then s > t and s < t are both covered by the same logic.
    #    Actually, if S % M == 0, then (S - x + y) % M == 0 is the same as (y - x) % M == 0.
    #    In that case, any two distinct indices i, j give two pairs (i, j) and (j, i).
    #    Total is N * (N-1) // 2 * 2 / (N/count... wait)
    #    Let's use the property: for each x, we need y such that (S - x + y) % M == 0.
    #    y = (x - S) % M.
    #    The number of pairs is sum_{x} (count(x) * count((x - S) % M))
    #    BUT we must exclude cases where s == t.
    #    Since s != t, if x != y, all count(x)*count(y) pairs are valid.
    #    If x == y, we have count(x) * (count(x) - 1) pairs.
    
    # Let's re-evaluate:
    # For a fixed s, we need t such that:
    # If t > s: P_mod[t-1] = P_mod[s-1]
    # If t < s: P_mod[t-1] = (P_mod[s-1] - S) % M
    
    # Total pairs = Sum_{i=0 to N-1} [ (count of P_mod[i] in indices > i) + (count of (P_mod[i]-S)%M in indices < i) ]
    # This is equivalent to:
    # 0.5 * Sum_{x} [ count(x) * (count(x) - 1) ]  <-- where s < t and P_mod[s-1] == P_mod[t-1]
    # + Sum_{x} [ count(x) * count((x - S) % M) ] <-- where s > t and P_mod[t-1] == (x - S) % M
    # BUT if S % M == 0, then (x - S) % M == x, and the second term becomes Sum count(x)^2.
    # We must subtract the cases where s == t, which is Sum count(x).
    # So if S % M == 0: 0.5 * (Sum count(x)^2 - N) + (Sum count(x)^2 - N) = 1.5 * (Sum count(x)^2 - N)
    # Wait, if S % M == 0, then for any s, t: dist(s, t) % M == 0 iff P_mod[s-1] == P_mod[t-1].
    # For each group of size 'c', we have c*(c-1) pairs.
    
    S_mod = S % M
    
    if S_mod == 0:
        # All pairs (s, t) with P_mod[s-1] == P_mod[t-1] and s != t
        # For each remainder x, we have count(x) * (count(x) - 1) pairs.
        ans = sum(c * (c - 1) for x, c in counts.items())
    else:
        # For each s, we need t such that:
        # 1. t > s and P_mod[t-1] == P_mod[s-1]
        # 2. t < s and P_mod[t-1] == (P_mod[s-1] - S_mod) % M
        # Total = Sum_{x} [ count(x) * (count(x)-1)//2 + count(x) * count((x - S_mod) % M) ]
        # Note: since S_mod != 0, x != (x - S_mod) % M, so the two conditions are disjoint.
        term1 = sum(c * (c - 1) // 2 for x, c in counts.items())
        term2 = sum(c * counts[(x - S_mod) % M] for x, c in counts.items())
        ans = term1 + term2
        
    sys.stdout.write(str(ans) + '\n')

if __name__ == '__main__':
    solve()