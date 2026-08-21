The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, emphasizing the transformation of data streams. In this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, and `collections.Counter` combined with `sum` and a generator expression to count pairs that satisfy the modulo condition.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the remaining A_i
    # Since we cannot use loops, we use next() to consume the iterator
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums of distances: P[i] is distance from area 1 to area i+1
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1] + P[t-1])
    
    # We only need the prefix sums modulo M
    # We prepend 0 to represent the starting point (rest area 1)
    prefixes = list(accumulate(a, lambda x, y: (x + y) % m, initial=0))
    
    # The total distance around the lake modulo M
    total_dist_mod = prefixes[-1]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0
    # For s < t: (prefixes[t-1] - prefixes[s-1]) % M == 0  => prefixes[t-1] == prefixes[s-1]
    # For s > t: (total_dist_mod - prefixes[s-1] + prefixes[t-1]) % M == 0 
    #            => prefixes[s-1] - prefixes[t-1] == total_dist_mod (mod M)
    
    # Count occurrences of each prefix sum modulo M
    # Note: prefixes contains N+1 elements, but the N-th element is the total sum.
    # The rest areas are 1 to N, corresponding to prefixes[0] to prefixes[N-1].
    counts = Counter(prefixes[:n])
    
    # For a fixed remainder r, let c = counts[r].
    # Pairs (s, t) with s < t and prefixes[s-1] == prefixes[t-1] == r:
    # There are c * (c - 1) / 2 such pairs.
    # However, we need to consider s > t as well.
    
    # Let's evaluate the condition for any pair (s, t) where s != t:
    # If s < t: distance is (prefixes[t-1] - prefixes[s-1]) % M
    # If s > t: distance is (total_dist_mod - (prefixes[s-1] - prefixes[t-1])) % M
    
    # Let x = prefixes[s-1] and y = prefixes[t-1]
    # If s < t, we need (y - x) % M == 0  => y == x
    # If s > t, we need (total_dist_mod - (x - y)) % M == 0 => x - y == total_dist_mod (mod M)
    
    # Let's iterate over all possible remainders r1 and r2 (0 to M-1)
    # But since M is up to 10^6, a nested loop is too slow.
    # We can use the counts dictionary to calculate the sum.
    
    # For each r1 in counts:
    # 1. Pairs (s, t) with s < t and prefixes[s-1] == prefixes[t-1] == r1:
    #    Contribution: counts[r1] * (counts[r1] - 1) / 2
    # 2. Pairs (s, t) with s > t and prefixes[s-1] - prefixes[t-1] == total_dist_mod (mod M):
    #    Let r2 = prefixes[t-1]. Then r1 - r2 == total_dist_mod (mod M)
    #    => r2 = (r1 - total_dist_mod) % M
    #    Contribution: counts[r1] * counts[r2]
    #    Special case: if r1 == r2 (which happens if total_dist_mod == 0), 
    #    the s > t case also gives counts[r1] * (counts[r1] - 1) / 2.
    
    # To avoid loops, we use a generator expression inside sum()
    # We iterate over the unique remainders present in the prefixes
    unique_rs = counts.keys()
    
    # Part 1: s < t and prefixes[s-1] == prefixes[t-1]
    # This is sum(c * (c-1) // 2 for c in counts.values())
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: s > t and prefixes[s-1] - prefixes[t-1] == total_dist_mod (mod M)
    # We need to sum counts[r1] * counts[(r1 - total_dist_mod) % M]
    # But we must exclude the case where s == t (which is already handled by the logic)
    # If total_dist_mod == 0, then r1 == r2, and we get counts[r1] * counts[r1].
    # But we need s > t, so it's counts[r1] * (counts[r1] - 1) / 2.
    
    # If total_dist_mod == 0:
    #   The condition for s > t is the same as s < t.
    #   Total = 2 * sum(c * (c-1) // 2)
    # If total_dist_mod != 0:
    #   The condition for s > t is r2 = (r1 - total_dist_mod) % M.
    #   Since total_dist_mod != 0, r1 != r2.
    #   Total = sum(c * (c-1) // 2) + sum(counts[r1] * counts[(r1 - total_dist_mod) % M])
    
    # We can unify this:
    # For each r1, we look for r2 = (r1 - total_dist_mod) % M.
    # If r1 == r2, we add c*(c-1)//2.
    # If r1 != r2, we add counts[r1] * counts[r2].
    
    # However, the "s < t" and "s > t" are distinct sets of pairs.
    # Let's just calculate them separately.
    
    # For s > t:
    # We need (total_dist_mod - (prefixes[s-1] - prefixes[t-1])) % M == 0
    # => prefixes[s-1] - prefixes[t-1] == total_dist_mod (mod M)
    # => prefixes[t-1] == (prefixes[s-1] - total_dist_mod) % M
    
    # Let r1 = prefixes[s-1], r2 = prefixes[t-1]
    # We want to count pairs (s, t) such that s > t and r2 = (r1 - total_dist_mod) % M.
    # This is equivalent to summing counts[r1] * counts[r2] for all r1, 
    # but we must handle the s > t constraint.
    # Actually, for any two distinct indices i, j from {0, ..., N-1}:
    # Either (i < j) or (i > j).
    # If i < j, the distance is (P[j] - P[i]) % M.
    # If i > j, the distance is (Total - (P[i] - P[j])) % M.
    
    # Let's use the property:
    # Pair (i, j) with i < j is valid if P[j] == P[i] (mod M).
    # Pair (i, j) with i > j is valid if P[i] - P[j] == Total (mod M).
    
    # Let's compute:
    # 1. Count pairs (i, j) with i < j and P[i] == P[j] (mod M).
    #    This is sum(c * (c-1) // 2 for c in counts.values()).
    # 2. Count pairs (i, j) with i > j and P[i] - P[j] == Total (mod M).
    #    This is sum(counts[r] * counts[(r - total_dist_mod) % M] for r in unique_rs)
    #    Wait, if Total == 0, then r == (r - Total) % M, so we get counts[r]^2.
    #    But we need i > j, so it should be counts[r] * (counts[r] - 1) // 2.
    #    If Total != 0, then r != (r - Total) % M, so we get counts[r] * counts[r2].
    
    # Correct logic for s > t:
    # We need to count pairs (i, j) such that 0 <= j < i < N and P[i] - P[j] == Total (mod M).
    # This is equivalent to P[j] == (P[i] - Total) (mod M).
    # For a fixed i, the number of j < i satisfying this is the number of times 
    # (P[i] - Total) % M appeared in P[0]...P[i-1].
    
    # To avoid loops, we can use a trick with a running tally or 
    # realize that if Total != 0, the sum of counts[r] * counts[(r - Total) % M] 
    # over all r is exactly the number of pairs (i, j) with P[i] - P[j] == Total (mod M).
    # Since r != (r - Total) % M, i and j must be distinct.
    # For every pair {i, j}, one is larger. If P[i] - P[j] == Total, then 
    # either (i > j and dist is Total) or (i < j and dist is -Total).
    # Actually, if Total != 0, then P[i] - P[j] == Total and P[j] - P[i] == Total 
    # cannot both be true.
    # So sum(counts[r] * counts[(r - total_dist_mod) % M]) counts all pairs (i, j) 
    # such that P[i] - P[j] == Total (mod M).
    # For each such pair, either i > j (which is what we want) or i < j.
    # If i < j and P[i] - P[j] == Total, then the distance is (P[i] - P[j]) % M = Total.
    # But we wanted the distance to be 0 % M.
    # So if Total != 0, the pairs (i, j) with i < j and P[i] - P[j] == Total 
    # are NOT the ones we counted in Part 1.
    
    # Let's simplify:
    # A pair (s, t) is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (Total - (P[s-1] - P[t-1])) % M == 0
    
    # Condition 1: P[t-1] == P[s-1] (mod M)
    # Condition 2: P[s-1] - P[t-1] == Total (mod M)
    
    # If Total == 0 (mod M):
    # Condition 1 and 2 both become P[s-1] == P[t-1] (mod M).
    # Total pairs = 2 * sum(c * (c-1) // 2) = sum(c * (c-1))
    
    # If Total != 0 (mod M):
    # Condition 1: P[s-1] == P[t-1] (mod M)
    # Condition 2: P[s-1] - P[t-1] == Total (mod M)
    # These two conditions are mutually exclusive.
    #