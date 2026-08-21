The constraint to avoid explicit loops forces a shift from imperative style to a functional paradigm, emphasizing data transformations over state mutation. In Python, this encourages the use of high-level built-ins and comprehensions which are often more concise and can be more performant. For this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions to handle the modulo operations, and `collections.Counter` to efficiently count occurrences of remainders.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Prefix sums of distances from rest area 1
    # P[i] is the distance from area 1 to area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN]
    # We use accumulate to avoid for-loops
    P = list(accumulate([0] + A))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    # Let S = Total_Sum = P[N]
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # and (S - P[s-1] + P[t-1]) % M == 0 for s > t
    
    S = P[N]
    # Remainders of prefix sums modulo M
    # R = [P[0]%M, P[1]%M, ..., P[N-1]%M]
    R = [p % M for p in P[:N]]
    
    # Count occurrences of each remainder
    counts = Counter(R)
    
    # For s < t: P[t-1] % M == P[s-1] % M
    # For a remainder r that appears k times, there are k*(k-1)//2 pairs
    ans_s_lt_t = sum([k * (k - 1) // 2 for k in counts.values()])
    
    # For s > t: (S + P[t-1] - P[s-1]) % M == 0
    # P[s-1] % M == (S + P[t-1]) % M
    # Let r_t = P[t-1] % M and r_s = P[s-1] % M
    # We need r_s == (S + r_t) % M
    # We iterate over all possible remainders r and their counts
    # Note: s > t means we are looking for pairs (s, t) where s is the starting point
    # and t is the destination.
    
    # To avoid loops, we use a list comprehension to calculate the sum
    # for each remainder r present in the Counter.
    ans_s_gt_t = sum([counts[r] * counts.get((S + r) % M, 0) 
                      for r in counts])
    
    # Special case: if (S + r) % M == r, the above logic counts pairs where s=t
    # But the problem says s != t. 
    # However, the s > t logic naturally handles s and t as distinct indices 
    # if we are careful. 
    # Let's refine:
    # For a fixed t, we need s > t such that P[s-1] % M == (S + P[t-1]) % M.
    # This is harder without loops. Let's use the property:
    # Total pairs = sum_{r} (count[r] * count[(S + r) % M])
    # But this includes cases where s=t (if S%M == 0) and doesn't account for s > t.
    
    # Correct approach for s != t:
    # Let r_i = P[i] % M for i = 0...N-1
    # Pair (s, t) is valid if:
    # 1. s < t and r_{t-1} == r_{s-1}
    # 2. s > t and r_{s-1} == (S + r_{t-1}) % M
    
    # Let's use the Counter to calculate:
    # Part 1: sum(k*(k-1)//2) for k in counts.values()
    # Part 2: sum(counts[r] * counts[(S + r) % M]) 
    # But Part 2 includes pairs where s=t if S%M == 0.
    # Also, Part 2 as written doesn't distinguish s > t.
    
    # Let's reconsider:
    # We want pairs (s, t) with s != t such that dist(s, t) % M == 0.
    # dist(s, t) = (P[t-1] - P[s-1]) if s < t else (S - P[s-1] + P[t-1])
    # In both cases, dist(s, t) = (P[t-1] - P[s-1]) % S
    # Actually, dist(s, t) % M = (P[t-1] - P[s-1]) % M if s < t
    # and dist(s, t) % M = (S + P[t-1] - P[s-1]) % M if s > t.
    
    # Let's use a different approach:
    # For each i in 0...N-1, let r_i = P[i] % M.
    # We want pairs (i, j) with i != j such that:
    # if i < j: (r_j - r_i) % M == 0  => r_j == r_i
    # if i > j: (S + r_j - r_i) % M == 0 => r_i == (S + r_j) % M
    
    # Total = sum_{i < j} [r_i == r_j] + sum_{i > j} [r_i == (S + r_j) % M]
    # The first term is sum(k*(k-1)//2)
    # The second term:
    # Let's use the fact that sum_{i > j} [r_i == (S + r_j) % M] 
    # is equivalent to saying for each j, count i > j such that r_i == (S + r_j) % M.
    # This is still loop-like. 
    
    # Alternative for the second term:
    # Let r'_j = (S + r_j) % M. We want pairs (i, j) with i > j and r_i == r'_j.
    # This is sum_{r} (count of r in R) * (count of r' in R) 
    # MINUS the cases where i < j and r_i == r'_j
    # MINUS the cases where i == j and r_i == r'_j.
    
    # This is getting complex. Let's use the simplest observation:
    # For every pair (s, t) with s != t, 
    # if S % M == 0, then dist(s, t) % M == (P[t-1] - P[s-1]) % M regardless of s < t.
    # If S % M != 0, then for a fixed s, t, only one of (s,t) or (t,s) can be 0 mod M
    # UNLESS (P[t-1] - P[s-1]) % M == 0 AND (S + P[t-1] - P[s-1]) % M == 0,
    # which implies S % M == 0.
    
    # So if S % M == 0:
    # The condition is simply r_{t-1} == r_{s-1}.
    # For each remainder r with count k, there are k*(k-1) pairs.
    # If S % M != 0:
    # We need r_{t-1} == r_{s-1} (for s < t) OR r_{s-1} == (S + r_{t-1}) % M (for s > t).
    # These two conditions are mutually exclusive.
    # Total = sum_{r} (count[r] * (count[r]-1)//2) + sum_{r} (count[r] * count[(S + r) % M])
    # Wait, the second term sum_{r} (count[r] * count[(S + r) % M]) 
    # counts pairs (s, t) where r_s == (S + r_t) % M.
    # Since S % M != 0, r_s cannot be equal to (S + r_s) % M, so s != t is guaranteed.
    # And since r_s == (S + r_t) % M and S % M != 0, it's impossible that r_s == r_t.
    # Thus the condition s > t is automatically satisfied if we just look for r_s == (S + r_t) % M
    # and we know r_s != r_t.
    # Actually, if r_s == (S + r_t) % M and r_s != r_t, then for the pair {s, t},
    # exactly one of (s, t) or (t, s) will satisfy the condition.
    
    # Let's refine:
    # If S % M == 0:
    #   Answer is sum(k * (k - 1) for k in counts.values())
    # If S % M != 0:
    #   Answer is sum(k * (k - 1) // 2 for k in counts.values()) + sum(counts[r] * counts[(S + r) % M] for r in counts)
    #   Wait, the second term: for any pair {s, t}, if r_s == (S + r_t) % M, 
    #   then r_t == (r_s - S) % M == (r_s + (M - (S % M))) % M.
    #   This means for any pair {s, t}, at most one of them can satisfy the condition.
    #   The sum sum(counts[r] * counts[(S + r) % M]) counts all pairs (s, t) such that r_s == (S + r_t) % M.
    #   Since S % M != 0, r_s != r_t, so s != t is guaranteed.
    #   Does this cover s > t? 
    #   If r_s == (S + r_t) % M, then dist(s, t) = (S + P[t-1] - P[s-1]) % M = 0.
    #   This is the condition for s > t.
    #   What if s < t? Then dist(s, t) = (P[t-1] - P[s-1]) % M.
    #   If r_s == (S + r_t) % M, then (P[t-1] - P[s-1]) % M = (r_t - (S + r_t)) % M = (-S) % M.
    #   Since S % M != 0, this is not 0.
    #   So the second term correctly counts all pairs (s, t) with s > t.
    
    if S % M == 0:
        print(sum([k * (k - 1) for k in counts.values()]))
    else:
        term1 = sum([k * (k - 1) // 2 for k in counts.values()])
        term2 = sum([counts[r] * counts.get((S + r) % M, 0) for r in counts])
        # Wait, the logic for term2: we want r_s == (S + r_t) % M.
        # This is exactly what sum(counts[r_s] * counts[r_t]) does.
        # But we need to be careful: the condition is r_s == (S + r_t) % M.
        # Let's use: sum(counts[r] * counts.get((r - S) % M, 0) for r in counts)
        # This counts pairs (s, t) such that r_s == (S + r_t) % M.
        # Let's double check: dist(s, t) = (S + P[t-1] - P[s-1]) % M = (S + r_t - r_s) % M.
        