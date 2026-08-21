```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map input to integers
    # N: number of rest areas, M: the divisor
    # A: list of steps between rest areas
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of A.
    # P_0 = 0
    # P_i = A_1 + ... + A_i
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1})
    # The distance from s to t (s > t) is (P_{N} - P_{s-1}) + P_{t-1}
    
    # We calculate prefix sums modulo M
    # Using accumulate to avoid explicit for-loops
    # We prepend 0 to handle the case where s=1
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    
    # Total distance around the lake modulo M
    total_sum_mod = P[N]
    
    # We need to find pairs (s, t) such that distance(s, t) % M == 0
    # Let x = P[s-1] and y = P[t-1]
    # If s < t: (y - x) % M == 0  => y % M == x % M
    # If s > t: (total_sum_mod - x + y) % M == 0 => (y - x) % M == -total_sum_mod % M
    
    # Count occurrences of each prefix sum modulo M
    counts = Counter(P[:N])
    
    # Case 1: s < t
    # For each value v that appears C times in P[0...N-1], 
    # we can pick 2 indices (s-1, t-1) in C*(C-1)//2 ways.
    ans_s_lt_t = sum(C * (C - 1) // 2 for C in counts.values())
    
    # Case 2: s > t
    # We need (y - x) % M == (-total_sum_mod) % M
    # Let target = (-total_sum_mod) % M
    # We need y % M == (x + target) % M
    # For each x, the number of y's is counts[(x + target) % M]
    # However, we must ensure s > t. 
    # This is tricky with counters. Let's use the property:
    # Total pairs (s, t) with s != t is N*(N-1).
    # Let's evaluate the condition (dist % M == 0) for all s, t.
    # For a fixed s, we need t such that:
    # If t > s: P[t-1] \equiv P[s-1] (mod M)
    # If t < s: P[t-1] \equiv P[s-1] - total_sum_mod (mod M)
    
    # Let's use a different approach:
    # For every pair (i, j) with 0 <= i < j < N:
    # Pair (i+1, j+1) is valid if P[j] - P[i] \equiv 0 (mod M)
    # Pair (j+1, i+1) is valid if P[N] - P[j] + P[i] \equiv 0 (mod M)
    
    # Let x = P[i], y = P[j]
    # Condition 1: y \equiv x (mod M)
    # Condition 2: y \equiv x + P[N] (mod M)
    
    # Let target = P[N] % M
    # If target == 0:
    #    Condition 1 and 2 are the same: y \equiv x (mod M).
    #    Each group of size C contributes C*(C-1) pairs.
    # If target != 0:
    #    Condition 1: y \equiv x (mod M) -> C*(C-1)//2 pairs
    #    Condition 2: y \equiv x + target (mod M) -> counts[x] * counts[(x + target)%M] pairs
    
    if total_sum_mod == 0:
        # Every pair (i, j) with P[i] == P[j] satisfies both directions
        # But we must exclude i == j.
        # For each value v, we have C(C-1) ordered pairs.
        result = sum(C * (C - 1) for C in counts.values())
    else:
        # s < t: P[t-1] == P[s-1]
        # s > t: P[t-1] == (P[s-1] - total_sum_mod) % M
        # These two conditions are mutually exclusive because total_sum_mod != 0.
        
        # Part 1: s < t
        term1 = sum(C * (C - 1) // 2 for C in counts.values())
        
        # Part 2: s > t
        # We need to count pairs (i, j) with 0 <= i < j < N such that 
        # P[i] == (P[j] - total_sum_mod) % M
        # This is equivalent to P[j] == (P[i] + total_sum_mod) % M
        # Since we can't use loops, we use a list comprehension and sum.
        # For each i, we need to know how many j > i satisfy the condition.
        # This is hard without loops. Let's use the global count instead.
        
        # Let's use the property: 
        # Total pairs (s, t) = \sum_{x \in \{0..M-1\}} (counts[x] * counts[(x - total_sum_mod) % M])
        # This counts all (s, t) such that (P[t-1] - P[s-1]) \equiv total_sum_mod (mod M)
        # Wait, the distance clockwise from s to t is:
        # If s < t: P[t-1] - P[s-1]
        # If s > t: P[N] - P[s-1] + P[t-1]
        
        # Let's re-evaluate:
        # Pair (s, t) is valid if:
        # 1. s < t AND P[t-1] \equiv P[s-1] (mod M)
        # 2. s > t AND P[t-1] \equiv P[s-1] - P[N] (mod M)
        
        # Let x = P[s-1] and y = P[t-1].
        # We want to count (x, y) from the set of prefix sums such that:
        # (index_x < index_y AND x == y) OR (index_x > index_y AND y == x - P[N])
        
        # Let's use the fact that:
        # \sum_{x, y} [x == y] = \sum C_i(C_i - 1) / 2 * 2 (if we ignore index)
        # Actually:
        # Count(s < t, P[s-1] == P[t-1]) = \sum C_i * (C_i - 1) // 2
        # Count(s > t, P[t-1] == P[s-1] - P[N]) = \sum_{i, j} [index_i > index_j AND P[j] == P[i] - P[N]]
        
        # Let's use a different trick.
        # For a fixed value v, let the indices be idx_{v,1}, idx_{v,2}, ...
        # For a fixed value w, let the indices be idx_{w,1}, idx_{w,2}, ...
        # We want to count (idx_{v, a}, idx_{w, b}) such that:
        # (idx_{v, a} < idx_{w, b} AND v == w) OR (idx_{v, a} > idx_{w, b} AND w == v - P[N])
        
        # If v == w: we get C_v * (C_v - 1) // 2 pairs.
        # If v != w: we need w == (v - P[N]) % M. 
        # Then we need to count pairs (idx_{v, a}, idx_{w, b}) with idx_{v, a} > idx_{w, b}.
        # This is still index dependent.
        
        # WAIT: The problem asks for the number of pairs (s, t).
        # Let's use the property:
        # For any two distinct indices i, j \in \{0, ..., N-1\},
        # one is smaller than the other.
        # Let i < j.
        # The clockwise distance from i+1 to j+1 is (P[j] - P[i]) % M.
        # The clockwise distance from j+1 to i+1 is (P[N] - P[j] + P[i]) % M.
        # We want (P[j] - P[i]) % M == 0 OR (P[N] - P[j] + P[i]) % M == 0.
        
        # Note: (P[j] - P[i]) % M == 0  <=>  P[j] == P[i] (mod M)
        # (P[N] - P[j] + P[i]) % M == 0 <=> P[j] - P[i] == P[N] (mod M)
        
        # If P[N] % M == 0, then both conditions are the same: P[j] == P[i].
        # Each pair {i, j} gives 2 pairs (s, t). Total = 2 * \sum C_i(C_i-1)//2 = \sum C_i(C_i-1).
        
        # If P[N] % M != 0, then the conditions P[j] == P[i] and P[j] - P[i] == P[N] are mutually exclusive.
        # For each pair {i, j} with i < j:
        # - If P[j] == P[i], then (i+1, j+1) is valid.
        # - If P[j] - P[i] == P[N], then (j+1, i+1) is valid.
        # - If P[i] - P[j] == P[N], then (i+1, j+1) is valid.
        
        # Total = \sum_{i < j} [P[j] == P[i]] + \sum_{i < j} [P[j] - P[i] == P[N]] + \sum_{i < j} [P[i] - P[j] == P[N]]
        # Total = \sum C_i(C_i-1)//2 + \sum_{i < j} [P[j] - P[i] == P[N] or P[i] - P[j] == P[N]]
        
        # Let target = P[N] % M.
        # We want to count pairs (i, j) with i < j such that P[j] - P[i] \equiv target (mod M) 
        # OR P[i] - P[j] \equiv target (mod M).
        # This is equivalent to:
        # \sum_{i < j} [P[j] \equiv P[i] + target] + \sum_{i < j} [P[j] \equiv P[i] - target]
        
        # To calculate \sum_{i < j} [P[j] \equiv P[i] + target] without loops:
        # We can use the fact that for any two values v and w, 
        # the number of pairs (i, j) with i < j such that P[i]=v and P[j]=w is:
        # (count of v) * (count of w) - (number of pairs i > j with P[i]=v and P[j]=w).
        # This is still complex.
        
        # Let's use the property:
        # For any two distinct indices i, j, exactly one of them is smaller.
        # Let the values be P[i] and P[j].
        # The pair is valid if (P[j] - P[i]) % M == 0 OR (P[i] - P[j]) % M == 0 (when P[N]=0)
        # OR (P