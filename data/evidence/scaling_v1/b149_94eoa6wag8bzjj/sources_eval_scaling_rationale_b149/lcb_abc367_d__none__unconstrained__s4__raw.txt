The constraint to avoid explicit loops forces a shift toward functional programming paradigms, which can lead to more concise and declarative code by focusing on "what" to compute rather than "how" to iterate. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, `collections.Counter` to count occurrences of remainders modulo $M$, and list comprehensions to handle the logic.

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
    
    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P_t - P_s)
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t)
    
    # Calculate prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + a[:-1]) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums.
    prefixes = list(accumulate([0] + a[:-1]))
    
    # We are interested in values modulo M
    mods = [p % m for p in prefixes]
    counts = Counter(mods)
    
    # Total distance around the lake
    total_sum = sum(a)
    total_mod = total_sum % m
    
    # For a pair (s, t):
    # If s < t: distance is (P_t - P_s). Condition: P_t % M == P_s % M
    # If s > t: distance is (Total - P_s + P_t). Condition: P_t % M == (P_s - Total) % M
    
    # Case 1: s < t
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need P_t % M == (P_s - total_mod) % M
    # Let r_t = P_t % M and r_s = P_s % M.
    # Condition: r_t == (r_s - total_mod) % m
    # This is equivalent to r_s == (r_t + total_mod) % m
    # For each r_t, the number of s > t is the count of r_s = (r_t + total_mod) % m
    # However, we must exclude cases where s = t (not allowed) and 
    # we must handle the "s > t" logic carefully.
    
    # A simpler way to think about s > t:
    # For every pair (s, t) with s < t, we checked if (P_t - P_s) % M == 0.
    # For the mirror pair (t, s) where t < s, the distance is (Total - P_s + P_t).
    # This is 0 mod M if P_s % M == (P_t + Total) % M.
    
    # Let's calculate the number of pairs (s, t) with s < t such that 
    # (Total + P_t - P_s) % M == 0.
    # This is P_s % M == (P_t + Total) % M.
    
    # To compute this without loops, we can use a generator expression:
    # For each possible remainder r, if we treat it as P_t % M, 
    # we need P_s % M to be (r + total_mod) % M.
    # But we need s > t. This is tricky with just counts.
    
    # Let's redefine:
    # We want pairs (s, t) such that dist(s, t) % M == 0.
    # dist(s, t) = (P_t - P_s) mod Total
    # If s < t: (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # If s > t: (Total + P_t - P_s) % M == 0 => P_s % M == (P_t + Total) % M
    
    # Let's use the property:
    # Total pairs = Sum_{r=0 to M-1} (count[r] * count[(r + total_mod) % M])
    # This sum includes cases where s < t, s > t, and s = t.
    # Specifically, if total_mod == 0, it counts s=t.
    # If total_mod == 0, then (P_t - P_s) % M == 0 and (Total + P_t - P_s) % M == 0 are the same.
    # But the problem says s != t.
    
    # Correct Logic:
    # For each pair i, j (i != j), the clockwise distance is:
    # if i < j: P_j - P_i
    # if i > j: Total + P_j - P_i
    
    # Let's evaluate Sum_{i=1 to N} Sum_{j=1 to N, j!=i} [dist(i, j) % M == 0]
    # = Sum_{i < j} [P_j - P_i % M == 0] + Sum_{i > j} [Total + P_j - P_i % M == 0]
    # = Sum_{i < j} [P_j % M == P_i % M] + Sum_{j < i} [P_i % M == (P_j + Total) % M]
    
    # Let's use the counts of remainders.
    # The first term is Sum (c_r * (c_r - 1) / 2)
    # The second term: for a fixed j, we need i > j such that P_i % M == (P_j + Total) % M.
    # This is harder to do with just global counts because of the i > j constraint.
    
    # Wait, the second term is:
    # Sum_{j < i} [P_i % M == (P_j + Total) % M]
    # Let's use a different approach for the second term.
    # Let r_j = P_j % M. We want count of i > j such that r_i = (r_j + total_mod) % M.
    # This can be solved by iterating through the list and keeping track of counts.
    # But I can't use loops. I can use a trick with a custom function and reduce or a list comprehension.
    
    # Actually, there is a simpler way.
    # Total pairs = Sum_{r=0 to M-1} (count[r] * count[(r + total_mod) % M])
    # This sum counts all pairs (i, j) such that (P_j - P_i + Total * [i > j]) % M == 0.
    # Let's check:
    # If i < j, we need P_j - P_i = 0 mod M.
    # If i > j, we need Total + P_j - P_i = 0 mod M => P_i - P_j = Total mod M.
    # Let's test this:
    # Sum_{r} count[r] * count[(r + total_mod) % M]
    # = Sum_{r, k: k = (r + total_mod)%M} count[r] * count[k]
    # = Number of pairs (i, j) such that P_j % M == (P_i % M + total_mod) % M.
    
    # If total_mod == 0:
    # It counts pairs (i, j) where P_j % M == P_i % M.
    # This includes i < j, i > j, and i = j.
    # For a fixed r, there are c_r * c_r pairs.
    # Total = Sum(c_r^2). 
    # Subtract i=j cases: Sum(c_r).
    # Result = Sum(c_r^2 - c_r) = Sum(c_r * (c_r - 1)).
    # This matches: 2 * Sum(c_r * (c_r - 1) / 2).
    
    # If total_mod != 0:
    # It counts pairs (i, j) such that P_j % M == (P_i % M + total_mod) % M.
    # Does this cover both i < j and i > j?
    # Let's use the property:
    # For any pair {i, j} with i < j:
    # Clockwise i -> j is (P_j - P_i)
    # Clockwise j -> i is (Total + P_i - P_j)
    # We want (P_j - P_i) % M == 0 OR (Total + P_i - P_j) % M == 0.
    # Note that if (P_j - P_i) % M == 0, then (Total + P_i - P_j) % M == Total % M.
    # Since total_mod != 0, these two conditions are mutually exclusive.
    # So for every pair {i, j}, at most one of the two directions is a multiple of M.
    # The number of pairs (i, j) with i < j such that (P_j - P_i) % M == 0 is Sum(c_r * (c_r - 1) / 2).
    # The number of pairs (i, j) with i < j such that (Total + P_i - P_j) % M == 0 is:
    # P_j % M == (Total + P_i) % M.
    # This is exactly what Sum_{r} (count[r] * count[(r + total_mod) % M]) would count if we 
    # didn't have the i < j constraint? No.
    
    # Let's use the most robust method:
    # The number of pairs (s, t) is:
    # Sum_{s < t} [P_t % M == P_s % M] + Sum_{s > t} [P_t % M == (P_s - Total) % M]
    # = Sum_{r} (c_r * (c_r - 1) / 2) + Sum_{s > t} [P_t % M == (P_s - Total) % M]
    # The second term: Sum_{t < s} [P_t % M == (P_s - Total) % M]
    # Let'_s = (P_s - Total) % M. We want count of t < s such that P_t % M == '_s.
    # This can be solved by iterating and keeping a running count.
    # To avoid loops, I'll use a list comprehension with a mutable external counter.
    
    # Wait, the second term is:
    # Sum_{t < s} [P_t % M == (P_s - Total) % M]
    # Let's use a dictionary to store counts of P_t % M seen so far.
    
    # Since I cannot use loops, I will use a helper function with a list to store state
    # and map() or a list comprehension to iterate.
    
    state = Counter()
    def count_and_update(p_s):
        target = (p_s - total_mod) % m
        res = state[target]
        state[p_s % m] += 1
        return res

    # Use map to apply the function to all prefixes
    # Note: prefixes are P_1, ..., P_N
    results = list(map(count_and_update, prefixes))
    
    # The first term (s < t) is handled by the logic:
    # For a fixed t, count s < t such that P_s % M == P_t % M.
    # This is exactly what the logic above does if total_mod == 0.
    # But we need both.
    
    # Let's restart the logic:
    # For each t from 1 to N:
    #   1. Count s < t such that P_s % M == P