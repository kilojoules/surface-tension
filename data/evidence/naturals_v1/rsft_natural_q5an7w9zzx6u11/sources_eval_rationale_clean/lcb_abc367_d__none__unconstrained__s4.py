The constraint to avoid explicit loops forces a shift from imperative programming to a functional paradigm, emphasizing data transformations over state mutation. In this problem, we need to calculate prefix sums of the distances between rest areas and count pairs $(s, t)$ such that the distance between them is a multiple of $M$. This is equivalent to finding pairs of prefix sums that are congruent modulo $M$. I will use `itertools.accumulate` for prefix sums, `map` for modulo operations, and `collections.Counter` to group and count occurrences of each remainder.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    # and then convert the remaining generator to a list
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (clockwise) is:
    # If s < t: P_t - P_s
    # If s > t: (Total_Sum - P_s) + P_t
    
    # Calculate prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + a[:-1]) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums.
    prefixes = list(accumulate([0] + a[:-1]))
    
    # We want (P_t - P_s) % M == 0 for s < t
    # and (Total_Sum + P_t - P_s) % M == 0 for s > t
    
    total_sum = sum(a)
    
    # Count occurrences of each remainder modulo M
    counts = Counter(map(lambda x: x % m, prefixes))
    
    # For s < t:
    # We need P_s % M == P_t % M.
    # For each remainder r, if there are c items, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If s < t, we have c*(c-1)//2 pairs.
    # If s > t, we need (Total_Sum + P_t - P_s) % M == 0, 
    # which means P_s % M == (P_t + Total_Sum) % M.
    
    # Let r_s = P_s % M and r_t = P_t % M.
    # Case 1: s < t => r_s == r_t
    # Case 2: s > t => r_s == (r_t + total_sum) % M
    
    # Total pairs = sum(count[r] * (count[r] - 1) // 2) for s < t
    # Plus sum(count[r_s] * count[r_t]) for s > t where r_s == (r_t + total_sum) % M
    # Note: s != t is guaranteed.
    
    # Calculation for s < t:
    # Using a generator expression and sum()
    ans_lt = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Calculation for s > t:
    # We need to iterate over all possible remainders r_t and find r_s.
    # r_s = (r_t + total_sum) % m
    # The number of pairs is count[r_t] * count[(r_t + total_sum) % m]
    # But we must exclude the case where s == t (which is already handled by s > t)
    # Wait, the condition s > t is strict. 
    # For a fixed r_t, any s that satisfies the condition and is "after" t in the 
    # linear prefix array is a valid pair.
    # This is tricky without loops. Let's rethink.
    
    # Let's use the property:
    # Total pairs = sum_{s=1 to N} sum_{t=1 to N, t!=s} [dist(s,t) % M == 0]
    # dist(s,t) = (P_t - P_s) % Total_Sum
    # Clockwise distance from s to t is (P_t - P_s) if t > s else (Total_Sum + P_t - P_s)
    # In both cases, clockwise distance is (P_t - P_s) mod Total_Sum.
    # We want (P_t - P_s) % M == 0, given that we are moving clockwise.
    # This is simply P_t % M == P_s % M IF we only consider the distance 
    # as the sum of A_i.
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # That is exactly (P_t - P_s) mod Total_Sum.
    # But the condition is simply that this value is a multiple of M.
    # (P_t - P_s) mod Total_Sum is a multiple of M if:
    # 1. t > s and (P_t - P_s) is a multiple of M
    # 2. t < s and (Total_Sum + P_t - P_s) is a multiple of M
    
    # Let r_i = P_i % M.
    # Condition 1: r_t == r_s
    # Condition 2: r_s == (r_t + Total_Sum) % M
    
    # Let T = Total_Sum % M.
    # If T == 0:
    # Condition 1 and 2 both become r_t == r_s.
    # For each r, we have c*(c-1) pairs (since s != t).
    # If T != 0:
    # Condition 1: r_t == r_s (for t > s)
    # Condition 2: r_s == (r_t + T) % M (for t < s)
    
    # Let's calculate:
    # For every pair (s, t) with s < t:
    #   Check if r_s == r_t
    # For every pair (s, t) with s > t:
    #   Check if r_s == (r_t + T) % M
    
    # This is equivalent to:
    # Sum_{r} (count[r] * (count[r]-1) // 2)  <-- for s < t
    # + Sum_{r} (count[(r + T) % M] * count[r]) <-- for s > t, but this includes s=t if T=0
    # Actually, for s > t, we can just iterate over all r and multiply count[r] by count[(r+T)%M].
    # But we must be careful: the "s > t" condition is about indices.
    # Let's use the identity:
    # Total = Sum_{s < t} [r_s == r_t] + Sum_{s > t} [r_s == (r_t + T) % M]
    
    # Let's use a different approach:
    # For each s, we want to count t != s such that dist(s,t) % M == 0.
    # dist(s,t) = (P_t - P_s) % Total_Sum.
    # (P_t - P_s) % Total_Sum is a multiple of M iff (P_t - P_s) is a multiple of M
    # PROVIDED that Total_Sum is also a multiple of M.
    # If Total_Sum is NOT a multiple of M, then (P_t - P_s) % Total_Sum 
    # is not simply (P_t - P_s) % M.
    
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # That is:
    # If s < t: P_t - P_s
    # If s > t: Total_Sum - (P_s - P_t) = Total_Sum + P_t - P_s
    
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # (s < t AND (P_t - P_s) % M == 0) OR (s > t AND (Total_Sum + P_t - P_s) % M == 0)
    
    # Let r_i = P_i % M and T = Total_Sum % M.
    # s < t: r_t == r_s
    # s > t: r_s == (r_t + T) % M
    
    # Let's use the counts of remainders.
    # For s < t, the number of pairs is sum(c*(c-1)//2).
    # For s > t, we can't use the total counts because the condition s > t depends on indices.
    # Let's use the fact that:
    # Sum_{s > t} [r_s == (r_t + T) % M] 
    # = Sum_{t < s} [r_s == (r_t + T) % M]
    
    # Let's process the prefixes and maintain a running count of remainders seen so far.
    # For each P_i:
    # 1. It can be a 't' for some s < i: count how many P_s % M == P_i % M
    # 2. It can be an 's' for some t < i: count how many P_t % M == (P_i - T) % M
    
    # Since we can't use loops, we can use a trick with a custom function and map/reduce
    # or just use the mathematical property:
    # Sum_{s < t} [r_s == r_t] = sum(c*(c-1)//2)
    # Sum_{s > t} [r_s == (r_t + T) % M] = ?
    
    # Let's use the property:
    # Sum_{s, t} [r_s == (r_t + T) % M] = Sum_{r} count[r] * count[(r + T) % M]
    # This sum includes:
    # 1. s < t and r_s == (r_t + T) % M
    # 2. s > t and r_s == (r_t + T) % M
    # 3. s = t and r_s == (r_s + T) % M (which is only true if T == 0)
    
    # This doesn't directly give Sum_{s > t}.
    # Let's go back to:
    # Ans = Sum_{s < t} [r_s == r_t] + Sum_{s > t} [r_s == (r_t + T) % M]
    
    # Let's use a list of counts of remainders and a way to sum them.
    # We can use a list comprehension to calculate the sum for s > t by 
    # iterating through the prefixes and using a frequency map.
    # But we can't use a loop to update the map.
    
    # Wait! We can use the fact that:
    # Sum_{s > t} [r_s == (r_t + T) % M] 
    # = Sum_{t < s} [r_s == (r_t + T) % M]
    # If we have the total counts of each remainder, 
    # Sum_{s, t} [r_s == (r_t + T) % M] = Sum_{r} count[r] * count[(r + T) % M]
    # And we know Sum_{s < t} [r_s == (r_t + T) % M] + Sum_{s > t} [r_s == (r_t + T) % M] 
    # + Sum_{s=t} [r_s == (r_s + T) % M] = Sum_{r} count[r] * count[(r + T) % M]
    
    # This still requires Sum_{s < t} [r_s == (r_t + T) % M].
    # Let's use the most direct way:
    # For s < t, we need r_s == r_t.
    # For s > t, we need r_s == (r_t + T) % M.
    
    # Let's use a list comprehension to build a list of "count of r_t seen so far