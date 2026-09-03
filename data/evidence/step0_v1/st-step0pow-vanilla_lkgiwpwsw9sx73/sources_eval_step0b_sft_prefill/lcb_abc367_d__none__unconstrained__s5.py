```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: number of rest areas, M: the divisor
    n = int(input_data[0])
    m = int(input_data[1])
    a = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances clockwise from rest area 1.
    # P[i] = distance from rest area 1 to rest area (i+1).
    # P[0] = 0, P[1] = A_1, P[2] = A_1 + A_2, ..., P[N] = sum(A_1...A_N)
    # We use accumulate to avoid explicit loops.
    p = list(accumulate(a, initial=0))
    
    # The distance clockwise from s to t is:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: (P[N] - P[s-1]) + P[t-1]
    # In both cases, we want (dist % M) == 0.
    # This is equivalent to P[t-1] % M == P[s-1] % M (mod M).
    
    # Let X_i = P[i] % M for i = 0, 1, ..., N-1.
    # We are looking for pairs (s, t) where 1 <= s, t <= N and s != t.
    # The condition is P[t-1] % M == P[s-1] % M.
    # Note: P[N] is the total circumference. If P[N] % M == 0, then 
    # P[t-1] - P[s-1] is a multiple of M iff P[t-1] % M == P[s-1] % M.
    # If P[N] % M != 0, the "wrap-around" distance is handled by the same logic:
    # (P[N] - P[s-1] + P[t-1]) % M == 0  =>  P[t-1] % M == (P[s-1] - P[N]) % M.
    
    # Let's simplify:
    # Let S = P[s-1] % M and T = P[t-1] % M.
    # If s < t: (T - S) % M == 0  => T == S
    # If s > t: (P[N] - S + T) % M == 0 => T == (S - P[N]) % M
    
    # Let L = P[N] % M.
    # For a fixed s, we need t such that:
    # 1. t > s and P[t-1] % M == P[s-1] % M
    # 2. t < s and P[t-1] % M == (P[s-1] - L) % M
    
    # Let's use a frequency map for all P[i] % M for i = 0...N-1.
    counts = Counter([val % m for val in p[:-1]])
    
    # Total pairs (s, t) with s != t such that dist(s, t) % M == 0.
    # For each s, we need t != s such that:
    # If s < t, P[t-1] % M = P[s-1] % M
    # If s > t, P[t-1] % M = (P[s-1] - L) % M
    
    # This is tricky to count without loops. Let's use the property:
    # Total = sum_{s=1}^N (count of t > s where P[t-1]%M == P[s-1]%M) 
    #       + sum_{s=1}^N (count of t < s where P[t-1]%M == (P[s-1]-L)%M)
    
    # Let X_i = P[i] % M for i = 0...N-1.
    # We want pairs (i, j) such that 0 <= i < j < N and X_j == X_i
    # PLUS pairs (i, j) such that 0 <= j < i < N and (X_i - X_j + L) % M == 0.
    # (X_i - X_j + L) % M == 0  => X_j == (X_i + L) % M.
    
    # Part 1: i < j and X_i == X_j.
    # For each value v, if it appears C_v times, there are C_v * (C_v - 1) // 2 pairs.
    # Part 2: j < i and X_j == (X_i + L) % M.
    # This is harder without a loop. But wait, we can use the total counts.
    # Let C_v be the count of X_k == v.
    # The number of pairs (i, j) with i != j such that dist(i, j) % M == 0 is:
    # For each i, we need X_j = (X_i + dist(i, j)) % M.
    # Actually, the condition is:
    # Clockwise distance from s to t is (P[t-1] - P[s-1]) mod P[N].
    # We want (P[t-1] - P[s-1]) % M == 0.
    # This is P[t-1] % M == P[s-1] % M.
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # If s < t, dist = P[t-1] - P[s-1].
    # If s > t, dist = P[N] - P[s-1] + P[t-1].
    # In both cases, dist % M == 0 is equivalent to:
    # (P[t-1] - P[s-1]) % M == 0  IF we consider the distance as a linear flow.
    # Let's re-verify:
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M.
    # If s > t: (P[N] - P[s-1] + P[t-1]) % M == 0 => P[t-1] % M == (P[s-1] - P[N]) % M.
    
    # Let X_i = P[i] % M. Let L = P[N] % M.
    # We want pairs (i, j) with 0 <= i, j < N, i != j such that:
    # 1. i < j and X_j == X_i
    # 2. i > j and X_j == (X_i - L) % M
    
    # Let's use the identity:
    # Total = sum_{i < j} [X_i == X_j] + sum_{i > j} [X_j == (X_i - L) % M]
    # Let C_v be the total count of value v in X.
    # sum_{i < j} [X_i == X_j] = sum_{v} C_v * (C_v - 1) // 2
    
    # For the second part: sum_{i > j} [X_j == (X_i - L) % M]
    # This is equivalent to counting pairs (j, i) with j < i such that X_j == (X_i - L) % M.
    # We can compute this by iterating through X and keeping track of counts of values seen so far.
    # To avoid loops, we can use a trick:
    # The total number of pairs (j, i) with j != i such that X_j == (X_i - L) % M is:
    # sum_{v} C_v * C_{(v + L) % M} (excluding cases where j=i).
    # If L == 0, this is sum C_v * (C_v - 1).
    # If L != 0, this is sum C_v * C_{(v + L) % M}.
    
    # But we only want j < i.
    # Notice that if we take all pairs (j, i) with j != i such that X_j == (X_i - L) % M,
    # and we also take all pairs (i, j) with i < j such that X_i == X_j,
    # this is not quite symmetric.
    
    # Let's use the property:
    # For a fixed pair {i, j} with i < j:
    # Clockwise i -> j is a multiple of M if X_j == X_i.
    # Clockwise j -> i is a multiple of M if X_i == (X_j - L) % M.
    
    # Total = sum_{i < j} ([X_i == X_j] + [X_i == (X_j - L) % M])
    # Total = sum_{v} (C_v * (C_v - 1) // 2) + sum_{i < j} [X_i == (X_j - L) % M]
    
    # To calculate sum_{i < j} [X_i == (X_j - L) % M] without a loop:
    # We can't easily. But we can use the fact that:
    # sum_{i < j} [X_i == (X_j - L) % M] + sum_{i > j} [X_i == (X_j - L) % M] 
    # = sum_{i != j} [X_i == (X_j - L) % M]
    # = sum_{v} C_{(v-L)%M} * C_v (minus cases where i=j, which is [X_i == (X_i - L)%M])
    
    # This is still not quite right because the "i < j" and "i > j" parts are different.
    # Let's use the "linear" approach:
    # Imagine the sequence X extended to 2N: X_0, X_1, ..., X_{N-1}, X_0, X_1, ..., X_{N-1}
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is P[N] - P[s-1] + P[t-1].
    # Both are multiples of M iff P[t-1] - P[s-1] is a multiple of M (mod P[N]).
    # Wait, the condition is simply: (P[t-1] - P[s-1]) % M == 0.
    # Because (P[N] - P[s-1] + P[t-1]) % M = (P[N] % M + (P[t-1] - P[s-1]) % M) % M.
    # For this to be 0, we need (P[t-1] - P[s-1]) % M == (-P[N]) % M.
    
    # Let X_i = P[i] % M.
    # Pair (s, t) is valid if:
    # 1. s < t and X_{t-1} == X_{s-1}
    # 2. s > t and X_{t-1} == (X_{s-1} - L) % M
    
    # Let's use the "complement" method.
    # For each i \in {0, ..., N-1}, we want to count j \in {0, ..., N-1}, j != i such that:
    # if i < j: X_j = X_i
    # if i > j: X_j = (X_i - L) % M
    
    # Let's process the array X and use a Fenwick tree or similar? No, that's for ranges.
    # We can use the fact that we can compute the "prefix" counts.
    # But we can also just use the total counts and the "diagonal" (i=j).
    # Let f(v) = count of X_k == v.
    # Total = sum_{i=0}^{N-1} (count of j > i with X_j == X_i) + sum_{i=0}^{N-1} (count of j < i with X_j == (X_i - L) % M)
    # The first term is sum_{v} f(v)*(f(v)-1)//2.
    