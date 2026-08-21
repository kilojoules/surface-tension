```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input and split into a list of integers
    data = list(map(int, sys.stdin.read().split()))
    
    # Extract N, M and the list of steps A
    N = data[0]
    M = data[1]
    A = data[2:]
    
    # Calculate prefix sums of A modulo M.
    # P[i] = (A_1 + ... + A_{i-1}) mod M.
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M.
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) mod M.
    # The distance from s to t (s > t) is (P[N] - P[s-1] + P[t-1]) mod M.
    
    # We use accumulate to get prefix sums and map them to their values modulo M.
    # We prepend 0 to handle the case where s=1.
    P = list(map(lambda x: x % M, accumulate([0] + A)))
    
    # Total distance around the lake modulo M
    total_sum_mod = P[N]
    
    # We need to find pairs (s, t) such that distance(s, t) % M == 0.
    # Case 1: s < t. 
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1].
    # Case 2: s > t.
    # (total_sum_mod - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == total_sum_mod (mod M).
    
    # Let's count occurrences of each value in P[0...N-1].
    # Note: P has N+1 elements, but we only care about the starting points of the N rest areas.
    # Rest area i is associated with P[i-1].
    counts = Counter(P[:N])
    
    # For Case 1 (s < t):
    # For each value v that appears k times, we have k*(k-1)//2 pairs.
    ans_case1 = sum(k * (k - 1) // 2 for k in counts.values())
    
    # For Case 2 (s > t):
    # We need P[s-1] - P[t-1] \equiv total_sum_mod (mod M).
    # This is equivalent to P[t-1] \equiv P[s-1] - total_sum_mod (mod M).
    # For each s, we need to count t < s that satisfy this.
    # However, it's easier to iterate through all possible values of P[s-1].
    # For a fixed value v1 = P[s-1] and v2 = P[t-1], the condition is (v1 - v2) % M == total_sum_mod.
    # This means v2 = (v1 - total_sum_mod) % M.
    # The number of pairs is sum(count[v1] * count[v2]) for all v1, v2 such that v1 != v2.
    # If v1 == v2, then total_sum_mod must be 0. But if total_sum_mod is 0, then v1=v2 is already handled in Case 1?
    # No, Case 1 is s < t, Case 2 is s > t.
    # If total_sum_mod == 0, then P[s-1] == P[t-1] implies distance is 0 mod M regardless of whether s < t or s > t.
    
    # Let's use a more robust approach for Case 2:
    # For every pair of distinct indices (i, j) where 0 <= i, j < N:
    # distance(i+1, j+1) is (P[j] - P[i]) % M.
    # We want (P[j] - P[i]) % M == 0, which is P[j] == P[i].
    # But the problem says s != t.
    # If we have k elements with value v, there are k*(k-1) ordered pairs (i, j) with P[i]=P[j].
    # These are exactly the pairs (s, t) where the clockwise distance is a multiple of M.
    
    # Wait, the clockwise distance from s to t is:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: (P[N] - P[s-1]) + P[t-1]
    # Both are equivalent to (P[t-1] - P[s-1]) mod M.
    
    # Proof:
    # If s < t, dist = P[t-1] - P[s-1].
    # If s > t, dist = (P[N] - P[s-1]) + P[t-1] = P[N] + (P[t-1] - P[s-1]).
    # For dist to be 0 mod M:
    # If s < t: P[t-1] \equiv P[s-1] (mod M)
    # If s > t: P[t-1] - P[s-1] \equiv -P[N] (mod M)
    
    # Let v1 = P[s-1] and v2 = P[t-1].
    # We need:
    # 1. s < t and v2 == v1
    # 2. s > t and v2 == (v1 - total_sum_mod) % M
    
    # Let's use the property:
    # Total pairs = (pairs with v2 == v1) + (pairs with v2 == (v1 - total_sum_mod) % M)
    # But we must exclude cases where s = t.
    # If total_sum_mod == 0, then (v1 - total_sum_mod) % M == v1.
    # In that case, we have two conditions that are identical.
    # If total_sum_mod != 0, the two conditions are mutually exclusive.
    
    # Correct Logic:
    # For each i \in {0, ..., N-1}, we look for j \in {0, ..., N-1} such that j != i and:
    # If i < j: P[j] == P[i]
    # If i > j: P[j] == (P[i] - total_sum_mod) % M
    
    # This is equivalent to:
    # Count pairs (i, j) with i < j and P[i] == P[j]
    # PLUS count pairs (i, j) with i > j and P[j] == (P[i] - total_sum_mod) % M
    
    # Let's use the Counter.
    # For a fixed value v, let count[v] be the number of times it appears in P[0...N-1].
    # The number of pairs (i, j) with i < j and P[i] == P[j] is sum(count[v] * (count[v]-1) // 2).
    # The number of pairs (i, j) with i > j and P[j] == (P[i] - total_sum_mod) % M:
    # This is trickier because of the i > j constraint.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t, the clockwise distance is a multiple of M iff:
    # P[t-1] - P[s-1] \equiv 0 (mod M) when s < t
    # P[t-1] - P[s-1] \equiv -P[N] (mod M) when s > t
    
    # Let's iterate through all i from 0 to N-1.
    # We want to count j such that:
    # (j > i and P[j] == P[i]) OR (j < i and P[j] == (P[i] - total_sum_mod) % M)
    
    # We can maintain a running count of P[j] values seen so far.
    # For a fixed i:
    # 1. The number of j < i such that P[j] == (P[i] - total_sum_mod) % M.
    # 2. The number of j > i such that P[j] == P[i].
    
    # To get (2) without a loop, we know:
    # (Total count of P[i]) = (count of j < i with P[j] == P[i]) + 1 + (count of j > i with P[j] == P[i])
    # So (count of j > i with P[j] == P[i]) = counts[P[i]] - 1 - (count of j < i with P[j] == P[i])
    
    # Let current_counts be the counts of P[j] for j < i.
    # For each i:
    # ans += current_counts[(P[i] - total_sum_mod) % M]
    # ans += (counts[P[i]] - 1 - current_counts[P[i]])
    # current_counts[P[i]] += 1
    
    # We can implement this using a list comprehension and a dictionary (or list since M is 10^6).
    
    # To avoid loops, we can use the mathematical identity:
    # Total = Sum_{v} (count[v] * (count[v]-1) // 2)  <-- this is s < t and P[s-1] == P[t-1]
    #       + Sum_{v} (count[v] * count[(v - total_sum_mod) % M]) 
    #       - (count of v such that v == (v - total_sum_mod) % M)
    # The second term is s > t and P[t-1] == (P[s-1] - total_sum_mod) % M.
    # Wait, the second term is actually:
    # For each s, we need t < s such that P[t-1] = (P[s-1] - total_sum_mod) % M.
    # This is exactly the number of pairs (t, s) with t < s and P[t-1] = (P[s-1] - total_sum_mod) % M.
    
    # Let v1 = P[s-1] and v2 = P[t-1].
    # We want:
    # (s < t and v1 == v2) OR (s > t and v2 == (v1 - total_sum_mod) % M)
    
    # Let's use the property:
    # Total = \sum_{v} \binom{count[v]}{2} + \sum_{v} (count[v] * count[(v - total_sum_mod) % M])
    # But we must be careful if total_sum_mod == 0.
    # If total_sum_mod == 0:
    # The condition is (s < t and v1 == v2) OR (s > t and v2 == v1).
    # This is just all pairs (s, t) with s != t and P[s-1] == P[t-1].
    # Which is \sum count[v] * (count[v] - 1).
    
    # If total_sum_mod != 0:
    # The two conditions (s < t, v1 == v2) and (s > t, v2 == v1 - total_sum_mod) are disjoint.
    # The first part is \sum \binom{count[v]}{2}.
    # The second part: for each pair of values (v, (v - total_sum_mod)%M), 
    # we have some number of indices. This is tricky.
    
    # Let's go back to:
    # For each i from 0 to N-1:
    #   ans += current_counts[(P[i] - total_sum_mod) % M]
    #   current_counts[P[i]] += 1
    # This counts pairs (j, i) with j < i such that P[j] == (P[i] - total_sum_mod) % M.
    # This is exactly the s > t case (where s = i+1, t = j+1).
    
    # Then we add the s < t case:
    # ans += sum(k * (k - 1) // 2 for k in counts.