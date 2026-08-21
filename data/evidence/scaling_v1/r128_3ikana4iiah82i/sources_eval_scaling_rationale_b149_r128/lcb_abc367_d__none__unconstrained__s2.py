The constraint to avoid explicit loops forces a shift from imperative style to a functional paradigm, emphasizing data transformations over state mutation. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, and `collections.Counter` to efficiently count occurrences of remainders modulo $M$. The final result is computed using a generator expression within `sum()`, iterating over the frequency map of remainders.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums of distances: P[i] is distance from area 1 to area i+1
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # We use accumulate to get the prefix sums and prepend 0
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Prefix sums modulo M
    # We only need the first N prefix sums (from area 1 to 1, 1 to 2, ..., 1 to N)
    # Note: The distance from 1 to 1 is 0.
    prefixes = list(accumulate(a, lambda x, y: (x + y) % m, initial=0))
    
    # The total distance around the lake modulo M
    total_dist_mod = sum(a) % m
    
    # We are looking for pairs (s, t) such that dist(s, t) % M == 0
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # If s > t: (Total - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == Total % M
    
    # Count frequencies of each remainder modulo M
    counts = Counter(prefixes[:n])
    
    # For a fixed remainder r, let count(r) be the number of prefixes with that remainder.
    # 1. Pairs (s, t) with s < t:
    #    For each r, we can pick 2 indices out of count(r).
    #    Ways = sum(count(r) * (count(r) - 1) // 2)
    # 2. Pairs (s, t) with s > t:
    #    We need (P[s-1] - P[t-1]) % M == total_dist_mod
    #    This means P[s-1] % M == (P[t-1] + total_dist_mod) % M
    #    Ways = sum(count(r) * count((r + total_dist_mod) % m))
    #    HOWEVER, we must exclude the case where s == t (though the problem says s != t)
    #    and we must handle the case where total_dist_mod == 0 carefully.
    
    # If total_dist_mod == 0, then s > t is the same condition as s < t.
    # But the problem asks for pairs (s, t).
    # Let's use the property: dist(s, t) % M == 0
    # Let P[i] be the distance from area 1 to area i+1.
    # dist(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # dist(s, t) = (P[N] - P[s-1] + P[t-1]) % M if s > t
    
    # Let's evaluate the sum for all s, t in {1, ..., N} where s != t:
    # The condition is:
    # If s < t: P[t-1] % M == P[s-1] % M
    # If s > t: P[s-1] % M == (P[t-1] + total_dist_mod) % M
    
    # Let C be the Counter of P[0...N-1] % M.
    # Total = sum_{r} (C[r] * (C[r]-1)//2)  <-- for s < t
    #       + sum_{r} (C[r] * C[(r - total_dist_mod) % m]) <-- for s > t
    # Note: In the second sum, if total_dist_mod == 0, we are counting pairs (s, t) 
    # where P[s-1] == P[t-1]. This includes s == t, so we must subtract N.
    
    ans = sum(c * (c - 1) // 2 for c in counts.values()) + \
          sum(counts[r] * counts[(r - total_dist_mod) % m] for r in counts) - \
          (n if total_dist_mod == 0 else 0)
          
    # Wait, the second sum counts all pairs (s, t) such that P[s-1] - P[t-1] == total_dist_mod.
    # If total_dist_mod == 0, it counts pairs where P[s-1] == P[t-1], including s == t.
    # If total_dist_mod != 0, it cannot count s == t because 0 != total_dist_mod.
    # But the second sum counts ALL pairs (s, t) regardless of whether s > t or s < t.
    # The problem asks for s != t.
    # Let's re-evaluate:
    # For each pair {i, j} with i < j:
    #   Check if dist(i, j) % M == 0  => (P[j-1] - P[i-1]) % M == 0
    #   Check if dist(j, i) % M == 0  => (P[N] - P[j-1] + P[i-1]) % M == 0
    # These are:
    #   1. P[j-1] % M == P[i-1] % M
    #   2. P[j-1] % M == (P[i-1] + total_dist_mod) % M
    
    # Let's use a different approach to avoid confusion:
    # For every pair i < j, we check two conditions.
    # Total = sum_{i < j} [P[i-1] == P[j-1]] + sum_{i < j} [P[j-1] == (P[i-1] + total_dist_mod) % M]
    # First term: sum(c * (c-1) // 2 for c in counts.values())
    # Second term: 
    # We want to count pairs (i, j) with i < j such that P[j-1] % M == (P[i-1] + total_dist_mod) % M.
    # This is harder to do with just a Counter because of the i < j constraint.
    # Actually, we can use the fact that:
    # sum_{i < j} [P[j-1] == (P[i-1] + total_dist_mod) % M] 
    # + sum_{i > j} [P[j-1] == (P[i-1] + total_dist_mod) % M] 
    # + sum_{i == j} [P[j-1] == (P[i-1] + total_dist_mod) % M]
    # = sum_{r} (counts[r] * counts[(r + total_dist_mod) % m])
    
    # Let's use the most direct logic:
    # A pair (s, t) is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (P[N] - P[s-1] + P[t-1]) % M == 0
    
    # Let P[i] be the prefix sums for i = 0...N-1.
    # Condition 1: P[t-1] % M == P[s-1] % M (with s < t)
    # Condition 2: P[s-1] % M == (P[t-1] + total_dist_mod) % M (with s > t)
    
    # Let's use a different method to count Condition 2:
    # For a fixed r, let indices be idx_1, idx_2, ..., idx_k.
    # We want to count pairs (idx_a, idx_b) such that a > b and P[idx_a] % M == (P[idx_b] + total_dist_mod) % M.
    # This is equivalent to counting pairs (s, t) such that s > t and P[s-1] % M == (P[t-1] + total_dist_mod) % M.
    # This is exactly what the second sum in my previous logic was doing, 
    # provided we handle the s < t and s == t cases.
    
    # Correct logic:
    # Total = sum_{r} (counts[r] * (counts[r]-1) // 2)  <-- This covers all s < t where P[s-1] == P[t-1]
    #       + sum_{r} (counts[r] * counts[(r - total_dist_mod) % m]) <-- This covers all s, t where P[s-1] - P[t-1] == total_dist_mod
    # Wait, the second term counts all (s, t) such that P[s-1] - P[t-1] == total_dist_mod.
    # If s > t, this is exactly the condition for dist(s, t) % M == 0.
    # If s < t, this is P[s-1] - P[t-1] == total_dist_mod => P[t-1] - P[s-1] == -total_dist_mod.
    # This is NOT the condition for s < t.
    # If s == t, this is 0 == total_dist_mod.
    
    # So:
    # For s < t, the condition is P[t-1] % M == P[s-1] % M.
    # For s > t, the condition is P[s-1] % M == (P[t-1] + total_dist_mod) % M.
    
    # Let's use the property:
    # Total = sum_{r} (counts[r] * (counts[r]-1) // 2)  <-- All pairs {s, t} such that P[s-1] == P[t-1]
    #       + sum_{r} (counts[r] * counts[(r - total_dist_mod) % m]) <-- All pairs (s, t) such that P[s-1] - P[t-1] == total_dist_mod
    # But the second term includes pairs where s < t, s > t, and s == t.
    # Let's refine:
    # The number of pairs (s, t) with s != t such that dist(s, t) % M == 0 is:
    # sum_{s < t} [P[s-1] == P[t-1]] + sum_{s > t} [P[s-1] == (P[t-1] + total_dist_mod) % M]
    
    # Let's use the fact that:
    # sum_{s, t} [P[s-1] == (P[t-1] + total_dist_mod) % M] = sum_{r} (counts[r] * counts[(r - total_dist_mod) % m])
    # This sum includes:
    # 1. s > t: The pairs we want.
    # 2. s < t: Pairs where P[s-1] - P[t-1] == total_dist_mod.
    # 3. s == t: Pairs where 0 == total_dist_mod.
    
    # This is getting complex. Let's use a simpler observation:
    # For any pair {s, t} with s < t:
    # dist(s, t) % M = (P[t-1] - P[s-1]) % M
    # dist(t,