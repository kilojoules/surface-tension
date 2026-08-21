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
    prefix_sums = list(accumulate(a, initial=0))
    
    # The distance clockwise from s to t is:
    # 1. If s < t: P[t-1] - P[s-1]
    # 2. If s > t: (P[N] - P[s-1]) + P[t-1]
    # In both cases, we want (Distance) % M == 0.
    # This is equivalent to P[t-1] % M == P[s-1] % M (for s < t)
    # or (P[t-1] - P[s-1] + P[N]) % M == 0 (for s > t).
    
    # Let X_i = P[i] % M for i = 0 to N-1.
    # We want to count pairs (s, t) with 1 <= s, t <= N and s != t such that:
    # If s < t: X_{t-1} == X_{s-1}
    # If s > t: X_{t-1} == (X_{s-1} - X_N) % M
    
    x = [p % m for p in prefix_sums[:-1]]
    total_sum_mod = prefix_sums[n] % m
    
    # Count occurrences of each remainder
    counts = Counter(x)
    
    # For a fixed s, we need to find t such that:
    # 1. t > s and X_{t-1} == X_{s-1}
    # 2. t < s and X_{t-1} == (X_{s-1} - total_sum_mod) % m
    
    # Total pairs = Sum_{s=1 to N} (count of t > s where X_{t-1} == X_{s-1}) 
    #              + Sum_{s=1 to N} (count of t < s where X_{t-1} == (X_{s-1} - total_sum_mod) % m)
    
    # Let C(v) be the number of times remainder v appears in X.
    # The number of pairs (s, t) with s < t and X_{s-1} == X_{t-1} is:
    # Sum_{v=0 to M-1} C(v) * (C(v) - 1) // 2
    
    # The number of pairs (s, t) with s > t and X_{t-1} == (X_{s-1} - total_sum_mod) % m:
    # This is trickier because of the s > t constraint.
    # Let's use the property: 
    # Total = Sum_{s, t} [dist(s, t) % M == 0] - Sum_{s} [dist(s, s) % M == 0]
    # Wait, the constraint is s != t.
    # Let's use the "two-cycle" trick.
    # Consider the sequence X_0, X_1, ..., X_{N-1}, X_N, X_{N+1}, ..., X_{2N-1}
    # where X_{N+i} = (X_i + total_sum_mod) % m.
    # The distance from s to t (s < t) is X_{t-1} - X_{s-1}.
    # The distance from s to t (s > t) is X_{t-1} + total_sum_mod - X_{s-1}.
    # This is X_{t-1 + N} - X_{s-1}.
    # So we want X_j == X_i where i < j < i + N.
    
    # For each i from 0 to N-1, we want to count j in {i+1, ..., i+N-1} such that X_j == X_i.
    # This is equivalent to:
    # For each value v, if it appears C(v) times in X_0...X_{N-1},
    # and we look at the extended sequence of length 2N,
    # the total number of pairs is Sum_{v} (C(v) * (C(v) - 1) // 2) 
    # BUT we must exclude cases where j >= i + N.
    # Since j < i + N, and we only have N distinct starting points,
    # the only way j >= i + N is if j = i + N.
    # X_{i+N} = (X_i + total_sum_mod) % m.
    # So if total_sum_mod % m == 0, then X_{i+N} == X_i, which we must exclude.
    # If total_sum_mod % m != 0, then X_{i+N} != X_i, so no pairs are excluded.
    
    # Correct Logic:
    # For each s \in {1...N}, we seek t \in {1...N}, t \neq s.
    # If s < t, we need X_{t-1} - X_{s-1} \equiv 0 \pmod M  => X_{t-1} \equiv X_{s-1} \pmod M.
    # If s > t, we need X_{t-1} - X_{s-1} + total_sum_mod \equiv 0 \pmod M => X_{t-1} \equiv X_{s-1} - total_sum_mod \pmod M.
    
    # Let C(v) be the count of v in X.
    # Pairs (s, t) with s < t: \sum_{v=0}^{M-1} C(v)(C(v)-1)//2
    # Pairs (s, t) with s > t: 
    # We need to count (t, s) such that t < s and X_{t-1} \equiv X_{s-1} - total_sum_mod \pmod M.
    # Let Y_i = X_i. We want to count (i, j) such that 0 <= i < j <= N-1 and Y_i \equiv Y_j - total_sum_mod \pmod M.
    # This is equivalent to Y_j \equiv Y_i + total_sum_mod \pmod M.
    
    # Let's use a different approach:
    # For every pair (i, j) with 0 <= i < j <= N-1:
    # Clockwise i -> j: distance is X_j - X_i. Multiple of M if X_j == X_i.
    # Clockwise j -> i: distance is (X_N - X_j) + X_i. Multiple of M if X_i == (X_j - X_N) % M.
    
    # Total = \sum_{v=0}^{M-1} [C(v) * (C(v)-1) // 2] + \sum_{0 <= i < j <= N-1} [X_i == (X_j - total_sum_mod) % M]
    
    # To calculate the second term:
    # We can iterate through j from 0 to N-1 and maintain the counts of X_i seen so far.
    # However, we can also use the property:
    # \sum_{0 <= i < j <= N-1} [X_i == (X_j - total_sum_mod) % M]
    # If total_sum_mod == 0:
    #   The second term is also \sum C(v)(C(v)-1)//2.
    # If total_sum_mod != 0:
    #   The two conditions X_i == X_j and X_i == (X_j - total_sum_mod) % M are mutually exclusive.
    #   The total count is simply the number of pairs (i, j) with i != j such that 
    #   dist(i, j) is a multiple of M.
    #   This is \sum_{i=0}^{N-1} (count of t != i such that dist(i, t) % M == 0).
    #   For a fixed i, t is clockwise from i.
    #   If t > i, dist = X_t - X_i. If t < i, dist = X_t - X_i + X_N.
    #   Both are 0 mod M if X_t \equiv X_i + (1 if t < i else 0) * X_N \pmod M.
    
    # Let's use the property:
    # For each i \in {0...N-1}, we want to count j \in {0...N-1}, j != i, such that:
    # 1. j > i and X_j == X_i
    # 2. j < i and X_j == (X_i - total_sum_mod) % M
    
    # Total = \sum_{i=0}^{N-1} (count of j > i with X_j == X_i) + \sum_{i=0}^{N-1} (count of j < i with X_j == (X_i - total_sum_mod) % M)
    # The first term is \sum C(v)(C(v)-1)//2.
    # The second term:
    # Let's process the array X and keep track of counts in a map.
    # For each X_i, the number of j < i satisfying the condition is current_counts[(X_i - total_sum_mod) % M].
    
    # Since we can't use loops, we can use a trick with map/reduce or a list comprehension.
    # But we need the state of the counter. 
    # Actually, we can calculate the second term without a loop:
    # The second term is \sum_{0 <= i < j <= N-1} [X_i == (X_j - total_sum_mod) % M].
    # This is \sum_{v=0}^{M-1} (count of v in X) * (count of (v + total_sum_mod)%M in X)
    # MINUS the cases where i >= j.
    # But we can just use:
    # For each pair of values (v, w) such that v == (w - total_sum_mod) % M:
    # We want to count pairs (i, j) with i < j, X_i = v, X_j = w.
    # This is hard without loops.
    
    # Let's reconsider:
    # For each i, we want to count j such that (X_j - X_i) % M == 0 (for j > i) 
    # and (X_j - X_i + total_sum_mod) % M == 0 (for j < i).
    # This is equivalent to:
    # Count pairs (i, j) with 0 <= i < j <= N-1 such that X_i == X_j
    # PLUS count pairs (i, j) with 0 <= i < j <= N-1 such that X_i == (X_j - total_sum_mod) % M.
    
    # Let C(v) be the count of value v in X.
    # The first term is \sum C(v)(C(v)-1)//2.
    # The second term:
    # If total_sum_mod == 0, it's the same as the first term.
    # If total_sum_mod != 0, then X_i and (X_j - total_sum_mod)%M are different values.
    # The number of pairs (i, j) with i < j such that X_i = v and X_j = w (where v = (w - total_sum_mod)%M)
    # depends on the relative positions.
    
    # Wait, there is a much simpler way.
    # For any two distinct rest areas s and t, one is clockwise from the other.
    # Let the distance from s to t be D. The distance from t to s is (Total_Distance - D).
    # We want to count pairs (s, t) such that D is a multiple of M.
    # Note: D is a