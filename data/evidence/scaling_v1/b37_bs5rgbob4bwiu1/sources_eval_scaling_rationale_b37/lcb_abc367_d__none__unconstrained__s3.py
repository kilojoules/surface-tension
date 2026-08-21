The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state changes. While idiomatic Python often uses `for` loops for readability, utilizing `itertools` and `functools.reduce` can lead to concise, expression-oriented code that describes *what* to calculate rather than *how* to step through the data. For this problem, I will use `itertools.accumulate` to generate prefix sums and a dictionary-based counting approach (via a list comprehension and `sum()`) to calculate the pairs.

```python
import sys
from itertools import accumulate

def solve():
    # Read N and M from the first line
    # Read A_i values from the remaining input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2
    # ...
    # P_i = sum(A_j for j from 1 to i-1)
    # The distance from s to t (s < t) is (P_t - P_s).
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t).
    
    # Calculate prefix sums modulo M
    # accumulate([0], A) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums (P_1 to P_N)
    prefixes = list(accumulate([0] + A, lambda x, y: (x + y) % M))[:N]
    total_sum_mod = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    # Using a list as a frequency array since M <= 10^6
    counts = [0] * M
    # We can't use a for loop, so we use a trick with a list comprehension 
    # or map to populate the counts. However, since we need to mutate 
    # the counts array, we use a loop-free way to aggregate:
    # We use a dictionary/counter logic via a comprehension and sum.
    
    # To count frequencies without a for loop, we can use a dictionary 
    # combined with the fact that we can iterate over the prefixes.
    # But the most "functional" way to count in Python is using collections.Counter.
    from collections import Counter
    freq = Counter(prefixes)
    
    # For a pair (s, t):
    # If s < t: (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # If s > t: (Total - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total % M
    
    # Case 1: s < t
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    ans1 = sum(c * (c - 1) // 2 for c in freq.values())
    
    # Case 2: s > t
    # We need (P_s - P_t) % M == total_sum_mod
    # This is equivalent to P_t % M == (P_s - total_sum_mod) % M
    # We iterate over all possible remainders r that exist in our prefix set.
    ans2 = sum(freq[r] * freq[(r - total_sum_mod) % M] for r in freq)
    
    # Special handling for Case 2: 
    # The above sum counts pairs (s, t) where s > t.
    # However, if total_sum_mod == 0, then (P_s - P_t) % M == 0 is the same as P_s == P_t.
    # The condition s > t means we are looking for pairs of indices (i, j) with i > j.
    # If total_sum_mod == 0, ans2 will calculate sum(c*c), which includes i=j and i<j.
    # Actually, the most robust way is:
    # For every s, we need a t < s such that P_t % M == (P_s - total_sum_mod) % M.
    # This is exactly what the sum over freq does, BUT we must subtract 
    # the cases where s=t (which happens if total_sum_0 == 0).
    
    # Correct logic for s > t:
    # We want pairs (s, t) with 1 <= t < s <= N such that (Total + P_t - P_s) % M == 0.
    # This is P_s - P_t \equiv Total (mod M).
    # Let target = total_sum_mod. We want P_s - P_t \equiv target (mod M).
    # For a fixed s, we need P_t \equiv (P_s - target) (mod M) where t < s.
    # This is tricky without a loop. Let's use the property:
    # Total pairs (s, t) with s != t is:
    # Sum_{r=0 to M-1} (count[r] * count[(r - total_sum_0) % M])
    # If total_sum_0 == 0, this counts s=t cases (count[r]) and both s<t and s>t.
    # If total_sum_0 != 0, it counts all pairs (s, t) such that P_s - P_t \equiv total_sum_0.
    # Since P_s - P_t \equiv total_sum_0 and P_t - P_s \equiv total_sum_0 cannot both be true
    # unless 2*total_sum_0 \equiv 0 (mod M), we can't just divide by 2.
    
    # Let's use the property:
    # Total = (Number of pairs (s, t) with s < t and P_s == P_t) 
    #       + (Number of pairs (s, t) with s > t and P_s - P_t == Total)
    
    # Let's redefine:
    # Valid pairs (s, t) are those where (P_t - P_s) % M == 0 for s < t
    # AND (Total + P_t - P_s) % M == 0 for s > t.
    
    # Part 1: s < t => P_t % M == P_s % M
    # This is sum(c*(c-1)//2)
    
    # Part 2: s > t => P_t % M == (P_s - Total) % M
    # Let's iterate over all possible remainders r1 (for P_s) and r2 (for P_t).
    # We need (r1 - r2) % M == total_sum_mod.
    # This means r2 = (r1 - total_sum_mod) % M.
    # For every pair of indices (i, j) with i > j, we check this.
    # This is equivalent to:
    # Sum_{r1} (count[r1] * count[(r1 - total_sum_mod) % M])
    # BUT this counts all pairs (i, j) regardless of whether i > j or i < j.
    # Wait, the condition (P_t - P_s) % M == 0 for s < t is DIFFERENT from
    # (Total + P_t - P_s) % M == 0 for s > t.
    
    # Let's use the property:
    # A pair (s, t) is valid if:
    # 1. s < t and P_t - P_s \equiv 0 (mod M)
    # 2. s > t and P_t - P_s \equiv -Total (mod M)
    
    # Let f(r) be the number of i such that P_i \equiv r (mod M).
    # Number of pairs (s, t) with s < t and P_s \equiv P_t \equiv r is f(r)*(f(r)-1)//2.
    # Number of pairs (s, t) with s > t and P_s \equiv r, P_t \equiv (r - Total) % M:
    # This is harder because of the s > t constraint.
    # Actually, let's use the fact that:
    # (s, t) is valid if (P_t - P_s) % M == 0 (for s < t)
    # OR (P_t - P_s + Total) % M == 0 (for s > t).
    # Note that (P_t - P_s + Total) % M == 0 is equivalent to (P_s - P_t) % M == Total % M.
    
    # Let's use a different approach:
    # For every pair i < j, they contribute to the answer if:
    # 1. P_i \equiv P_j (mod M)  -- this is the pair (s=i, t=j)
    # 2. P_j - P_i \equiv Total (mod M) -- this is the pair (s=j, t=i)
    
    # Total = Sum_{r} [f(r)*(f(r)-1)//2] + Sum_{i < j} [P_j - P_i \equiv Total (mod M)]
    # The second term: Sum_{i < j} [P_j - P_i \equiv Total (mod M)]
    # This can be solved by iterating through the array and keeping track of counts.
    # Since we can't use loops, we can use a custom function with reduce or a list comprehension.
    
    # To calculate Sum_{i < j} [P_j - P_i \equiv Total (mod M)]:
    # We need P_i \equiv (P_j - Total) (mod M).
    # We can use a list comprehension to simulate the accumulation of counts.
    # However, the most Pythonic way to do this without 'for' is using a scan (accumulate).
    # But we need the counts of previous elements.
    
    # Let's use the property:
    # Sum_{i < j} [P_j - P_i \equiv Total] = 
    # (Sum_{r} f(r) * f((r - Total) % M) - Sum_{i} [P_i - P_i \equiv Total]) / 2 
    # ... No, that only works if Total == 0.
    
    # Correct approach for Sum_{i < j} [P_j - P_i \equiv Total]:
    # Let g(r) be the number of i such that P_i \equiv r (mod M).
    # The total number of pairs (i, j) with i != j such that P_j - P_i \equiv Total is:
    # Sum_{r} f(r) * f((r - Total) % M)
    # If Total == 0, this is Sum f(r)^2. We must subtract the i=j cases: Sum f(r).
    # Then we have pairs (i, j) with i != j.
    # For every such pair, either (i < j and P_j - P_i \equiv Total) 
    # or (i > j and P_j - P_i \equiv Total).
    # This doesn't help because we specifically need i < j for the second term.
    
    # WAIT! The problem says:
    # "The minimum number of steps required to walk clockwise from s to t is a multiple of M."
    # If s < t, distance is P_t - P_s.
    # If s > t, distance is (Total + P_t - P_s).
    # Both conditions are: (P_t - P_s) % M == 0 (if s < t) OR (P_t - P_s + Total) % M == 0 (if s > t).
    # This is equivalent to:
    # Pair (s, t) is valid if:
    # 1. s < t AND P_s \equiv P_t (mod M)
    # 2. s > t AND P_s - P_t \equiv Total (mod M)
    
    # Let's use the fact that:
    # Sum_{s < t} [P_s \equiv P_t] = Sum_{r} f(r)(f(r)-1)//2
    # Sum_{s > t} [P_s - P_t \equiv Total] = Sum_{s} (count of t < s such that P_t \equiv P_s - Total)
    
    # To calculate the