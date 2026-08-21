The constraint to avoid explicit loops forces a shift from imperative programming to a functional paradigm, emphasizing data transformations over state mutation. This approach encourages the use of higher-order functions and comprehensions, which can often be more concise and better optimized by the Python interpreter. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions for data processing, and `collections.Counter` to efficiently count occurrences of remainders modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = list(map(int, sys.stdin.read().split()))
    
    # N: number of rest areas, M: the divisor
    # A: list of distances between rest area i and i+1
    N, M = input_data[0], input_data[1]
    A = input_data[2:]
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Calculate prefix sums modulo M
    # accumulate provides P[1]...P[N]. We prepend 0 for P[0].
    P = [0] + list(accumulate(A, lambda x, y: (x + y) % M))
    
    # Total sum of all A_i modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # For s > t: (total_sum_mod - P[s-1] + P[t-1]) % M == 0 
    #            => P[t-1] % M == (P[s-1] - total_sum_mod) % M
    
    # Note: P[N] is the distance from 1 back to 1. 
    # The problem defines s != t.
    # Let's consider the remainders of P[0]...P[N-1]
    rems = P[:-1]
    counts = Counter(rems)
    
    # For a fixed s, we want to find t != s such that:
    # If t > s: P[t-1] % M == P[s-1] % M
    # If t < s: P[t-1] % M == (P[s-1] - total_sum_mod) % M
    
    # Let R = P[s-1] % M.
    # Number of t > s is (count of R in P[0...N-1]) - (count of R in P[0...s-1])
    # Number of t < s is (count of (R - total_sum_mod) % M in P[0...s-1])
    
    # However, it is simpler to think:
    # For every s, we seek t such that distance is 0 mod M.
    # Total pairs = Sum_{s=1 to N} (count of t such that dist(s,t) % M == 0)
    # Subtract cases where s == t (though the problem says s != t, 
    # the logic might include it if we aren't careful).
    
    # Let's use the property:
    # For each s, the required P[t-1] % M is:
    # 1. If t > s, P[t-1] % M = P[s-1] % M
    # 2. If t < s, P[t-1] % M = (P[s-1] - total_sum_mod) % M
    
    # Let's evaluate the contribution of each remainder r in counts:
    # For a remainder r, there are counts[r] indices.
    # For each such index s, the number of t > s with P[t-1] % M == r 
    # is (counts[r] - 1) distributed across all s.
    # Total pairs (s, t) with s < t and P[s-1] == P[t-1] is:
    # sum(v * (v - 1) // 2 for v in counts.values())
    # But we need to account for the clockwise wrap-around.
    
    # Let's redefine:
    # For each s in {1...N}, we want t in {1...N}, t != s such that:
    # If s < t: P[t-1] ≡ P[s-1] (mod M)
    # If s > t: P[t-1] ≡ P[s-1] - total_sum_mod (mod M)
    
    # Let r_s = P[s-1] % M.
    # The number of t > s such that P[t-1] ≡ r_s (mod M) is:
    # (count of r_s in P[0...N-1]) - (rank of s in indices of r_s)
    # The number of t < s such that P[t-1] ≡ r_s - total_sum_mod (mod M) is:
    # (count of (r_s - total_sum_mod) in P[0...s-2])
    
    # This is still loop-like. Let's use the aggregate counts:
    # For a fixed remainder r, let C(r) be the number of times it appears in P[0...N-1].
    # For each s where P[s-1] = r:
    #   Pairs (s, t) with s < t: there are (C(r) - 1) such t's in total across all s.
    #   Wait, the sum of (C(r) - 1) for all s is C(r)*(C(r)-1).
    #   But we only want t > s. That is C(r)*(C(r)-1)//2.
    #   Actually, for a fixed r, the number of pairs (s, t) with s < t and P[s-1]=P[t-1]=r 
    #   is C(r) * (C(r) - 1) // 2.
    #   For the wrap-around (s > t), we need P[t-1] ≡ P[s-1] - total_sum_mod (mod M).
    #   Let r_t = P[t-1] and r_s = P[s-1].
    #   We need r_t ≡ r_s - total_sum_mod (mod M), which is r_s ≡ r_t + total_sum_mod (mod M).
    #   For each t, the number of s > t such that r_s ≡ r_t + total_sum_mod (mod M) is:
    #   (Count of (r_t + total_sum_mod) % M in P[0...N-1]) - (Count of (r_t + total_sum_mod) % M in P[0...t-1])
    
    # Let's simplify:
    # Total pairs = Sum_{s < t} [P[s-1] == P[t-1]] + Sum_{s > t} [P[s-1] == P[t-1] + total_sum_mod]
    # Part 1: Sum_{r} C(r) * (C(r) - 1) // 2
    # Part 2: Sum_{t < s} [P[s-1] == (P[t-1] + total_sum_mod) % M]
    # For a fixed t, we need s > t such that P[s-1] == (P[t-1] + total_sum_mod) % M.
    # This is (Count of (P[t-1] + total_sum_mod) % M in P[0...N-1]) - (Count of (P[t-1] + total_sum_mod) % M in P[0...t-1])
    
    # Let r_t = P[t-1]. Let r_s = (r_t + total_sum_mod) % M.
    # Total Part 2 = Sum_{t=1 to N} (C(r_s) - Count of r_s in P[0...t-1])
    # Total Part 2 = Sum_{t=1 to N} C(r_s) - Sum_{t=1 to N} (Count of r_s in P[0...t-1])
    
    # Sum_{t=1 to N} C(r_s) = Sum_{r_t} C(r_t) * C((r_t + total_sum_mod) % M)
    # Sum_{t=1 to N} (Count of r_s in P[0...t-1]) is the sum of indices of r_s.
    # If r_s appears at indices i_1, i_2, ..., i_k (0-indexed),
    # the contribution to the sum is:
    # For t=1: count of r_s in P[0...0]
    # For t=2: count of r_s in P[0...1] ...
    # This is equivalent to Sum_{j=1 to k} (N - i_j)
    # Wait, let's re-evaluate:
    # Sum_{t=1 to N} (Count of r_s in P[0...t-1]) 
    # = Sum_{t=1 to N} Sum_{i=0 to t-1} [P[i] == r_s]
    # = Sum_{i=0 to N-1} Sum_{t=i+1 to N} [P[i] == r_s]
    # = Sum_{i=0 to N-1} [P[i] == r_s] * (N - i)
    
    # Let's use the property:
    # Total Part 2 = Sum_{t=1 to N} C((P[t-1] + total_sum_mod) % M) - Sum_{t=1 to N} (Count of (P[t-1] + total_sum_mod) % M in P[0...t-1])
    # The second term: Sum_{t=1 to N} Sum_{i=0 to t-1} [P[i] == (P[t-1] + total_sum_mod) % M]
    # = Sum_{i=0 to N-1} Sum_{t=i+1 to N} [P[i] == (P[t-1] + total_sum_mod) % M]
    # = Sum_{i=0 to N-1} Sum_{t=i+1 to N} [P[t-1] == (P[i] - total_sum_mod) % M]
    
    # Let r_i = P[i]. Let r_target = (r_i - total_sum_mod) % M.
    # The inner sum is the number of t > i+1 such that P[t-1] == r_target.
    # This is (C(r_target) - Count of r_target in P[0...i]).
    
    # Actually, the simplest way:
    # A pair (s, t) is valid if:
    # 1. s < t and P[t-1] ≡ P[s-1] (mod M)
    # 2. s > t and P[s-1] ≡ P[t-1] + total_sum_mod (mod M)
    
    # Let's use the counts of remainders.
    # For each remainder r, let S_r be the set of indices i where P[i] == r.
    # Part 1: For each r, combinations(len(S_r), 2)
    # Part 2: For each r, we need s > t such that P[s-1] == (P[t-1] + total_sum_mod) % M.
    # Let r_t = r and r_s = (r + total_sum_mod) % M.
    # We need to count pairs (t, s) such that t < s, P[t-1] = r_t, and P[s-1] = r_s.
    # If r_t == r_s (which happens if total_sum_mod == 0), this is just Part 1.
    # If r_t