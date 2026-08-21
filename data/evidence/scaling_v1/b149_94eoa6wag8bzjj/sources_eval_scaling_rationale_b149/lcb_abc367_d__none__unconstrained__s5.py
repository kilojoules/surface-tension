The constraint to avoid explicit loops forces a shift toward a functional programming style, emphasizing declarative data transformations over imperative state changes. This approach leverages Python's highly optimized internal C-loops via built-ins and comprehensions, which can often be more concise and idiomatic for data processing tasks. To solve this problem, I will use `itertools.accumulate` to compute prefix sums of the distances, list comprehensions for transformations, and `collections.Counter` to efficiently count occurrences of remainders modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the remaining A_i values
    # Since we cannot use loops, we use next() to grab N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P_t - P_s).
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t).
    # We want (Distance) % M == 0.
    
    # Compute prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + a[:-1]) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums.
    prefixes = list(accumulate([0] + a[:-1]))
    total_sum = sum(a)
    
    # We are looking for pairs (s, t) such that:
    # 1. s < t and (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # 2. s > t and (Total_Sum - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total_Sum % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(p % m for p in prefixes)
    
    # For s < t:
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    # This is summed across all r.
    ans_s_less_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For s > t:
    # We need (P_s - P_t) % M == Total_Sum % M
    # Let R = Total_Sum % M.
    # We need P_s % M - P_t % M == R (mod M)
    # Which is P_t % M == (P_s % M - R) % M
    r_total = total_sum % m
    
    # For each s, we need to count how many t < s satisfy the condition.
    # However, it's easier to iterate over the counts of remainders.
    # For a fixed remainder r_s, we need r_t = (r_s - r_total) % m.
    # The number of pairs is sum(count(r_s) * count(r_t)) for all r_s.
    # But we must exclude the case where s == t (though the problem says s != t).
    # The condition s > t is strictly required.
    # Actually, the total number of pairs (s, t) with s != t is:
    # Sum_{s, t} [dist(s, t) % M == 0]
    # Let's use the property: 
    # For a fixed s, we want t such that:
    # If t > s: P_t % M == P_s % M
    # If t < s: P_t % M == (P_s % M - Total_Sum) % M
    
    # Let's calculate the total pairs using the Counter:
    # For each remainder r, there are count(r) positions.
    # Each position s with P_s % M == r can pair with:
    # 1. Any t > s with P_t % M == r  (handled by c*(c-1)//2)
    # 2. Any t < s with P_t % M == (r - r_total) % m
    
    # To avoid loops and recursion, we use a generator expression:
    # For each r in counts, the number of t < s is tricky because it depends on indices.
    # Let's reconsider:
    # Total pairs = Sum_{s=1 to N} (count of t > s where P_t % M == P_s % M)
    #             + Sum_{s=1 to N} (count of t < s where P_t % M == (P_s % M - r_total) % M)
    
    # Let's use the fact that:
    # Sum_{s < t} [P_t % M == P_s % M] = sum(c*(c-1)//2 for c in counts.values())
    # Sum_{s > t} [P_t % M == (P_s % M - r_total) % M] 
    # = Sum_{r} (count(r) * count((r - r_total) % m)) 
    #   - Sum_{r} [r == (r - r_total) % m] * (count(r) * (count(r)+1) // 2) ... No.
    
    # Let's use a different approach for s > t:
    # We want pairs (s, t) with s > t and P_s % M - P_t % M \equiv r_total (mod M).
    # This is equivalent to counting pairs (r_s, r_t) from the distribution of remainders,
    # but we must handle the indices.
    # Actually, the simplest way is:
    # For every pair of remainders (r1, r2) such that (r1 - r2) % M == r_total:
    # If r1 != r2, there are count(r1) * count(r2) pairs.
    # Some have s < t, some have s > t.
    # But the condition for s < t is P_t % M == P_s % M (which means r_total must be 0).
    # If r_total == 0, then s < t and s > t both require P_s % M == P_t % M.
    # Total pairs = N * (N-1) if M=1, etc.
    
    # Correct Logic:
    # A pair (s, t) is valid if:
    # 1. s < t and (P_t - P_s) % M == 0
    # 2. s > t and (Total_Sum - P_s + P_t) % M == 0
    
    # Case 1: s < t
    # P_t % M == P_s % M. 
    # For each remainder r, if it appears c times, there are c*(c-1)//2 pairs.
    
    # Case 2: s > t
    # P_t % M == (P_s % M - Total_Sum) % M.
    # Let r_s = P_s % M and r_t = P_t % M.
    # We need r_t == (r_s - r_total) % M.
    # For each possible remainder r, we have count(r) choices for s and count((r - r_total) % M) choices for t.
    # This gives count(r) * count((r - r_total) % M) pairs.
    # However, this includes cases where s < t or s == t.
    # We specifically need s > t.
    
    # Let's use the property:
    # Total valid pairs = Sum_{s < t} [P_t % M == P_s % M] + Sum_{s > t} [P_t % M == (P_s % M - r_total) % M]
    # Let f(r) = count(r).
    # Sum_{s < t} [P_t % M == P_s % M] = Sum_{r} f(r)*(f(r)-1)//2
    # Sum_{s > t} [P_t % M == (P_s % M - r_total) % M]:
    # Let r_t = (r_s - r_total) % M.
    # If r_t != r_s:
    #   The number of pairs (s, t) with P_s % M = r_s and P_t % M = r_t is f(r_s)*f(r_t).
    #   Some have s < t, some have s > t.
    #   But the condition for s < t was P_t % M == P_s % M.
    #   Since r_t != r_s, all these f(r_s)*f(r_t) pairs must be checked for s > t.
    #   Wait, the condition s < t and s > t are mutually exclusive for a fixed pair {s, t}.
    #   For a fixed pair {s, t} with s < t:
    #   It is valid if P_t % M == P_s % M  OR  (Total_Sum - P_s + P_t) % M == 0.
    #   Note: (Total_Sum - P_s + P_t) % M == 0  <=>  (P_t - P_s) % M == -Total_Sum % M.
    
    # Let's just use the direct sum:
    # Ans = Sum_{s < t} [ (P_t - P_s) % M == 0 ] + Sum_{s < t} [ (Total_Sum - (P_t - P_s)) % M == 0 ]
    # Ans = Sum_{r} f(r)(f(r)-1)//2 + Sum_{s < t} [ (P_t - P_s) % M == Total_Sum % M ]
    
    # For the second term: (P_t - P_s) % M == r_total.
    # If r_total == 0, it's the same as the first term: Sum f(r)(f(r)-1)//2.
    # If r_total != 0, then P_t % M != P_s % M.
    # For any pair {s, t} with s < t, it's either:
    # 1. P_t % M == P_s % M
    # 2. P_t % M == (P_s % M + r_total) % M
    # 3. Neither.
    # These two conditions are mutually exclusive if r_total != 0.
    
    # If r_total == 0:
    # Ans = 2 * Sum_{r} f(r)(f(r)-1)//2
    
    # If r_total != 0:
    # Term 1: Sum_{r} f(r)(f(r)-1)//2
    # Term 2: Sum_{s < t} [ P_t % M == (P_s % M + r_total) % M ]
    # To compute Term 2 without loops:
    # We can use the fact that Sum_{s < t} [P_t % M == r_t and P_s % M == r_s] 
    # is not simply f(r_s)*f(r_t) because of the s < t constraint.
    # Actually, we can use a prefix-sum-like approach with a list comprehension.
    # But we can't use loops. We can use a trick with a custom class or a mutable object 
    # inside a list comprehension to maintain state.
    
    # Wait, there is a simpler way for Term 2:
    # Sum_{s < t} [P_t % M == (P_s % M + r_total) % M]
    # Let's use the property: 
    # Total pairs (s, t) with s != t such that (P_t - P_s) % M == r_total
    # is Sum_{r} f(r) * f((r + r_total) % M), but we must subtract cases where s=t.
    # Since r_total != 0, s=t is impossible.
    # This sum includes both s < t and s > t.
    # For a fixed pair {s, t} with s < t, the distance clockwise from s to t is (P_t - P_s).
    # The distance clockwise from t to s is (Total_Sum -