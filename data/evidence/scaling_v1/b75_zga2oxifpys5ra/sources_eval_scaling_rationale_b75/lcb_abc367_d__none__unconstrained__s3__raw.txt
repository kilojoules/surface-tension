The constraint to avoid explicit loops and recursion forces a shift from imperative programming to a functional style. While idiomatic Python often uses `for` loops for readability, utilizing `itertools` and `collections.Counter` allows for concise data transformations and aggregations. For this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions to handle modular arithmetic, and `collections.Counter` to count occurrences of each remainder, finally using a generator expression within `sum()` to calculate the total number of valid pairs.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Prefix sums of distances: P[i] is distance from rest area 1 to rest area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN]
    # We only need the first N prefix sums (0 to N-1)
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # P_mod[i] = (sum of A_0...A_{i-1}) % M
    P_mod = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # P_mod has N+1 elements. We only need the first N for the starting positions.
    # P_mod[0] = 0 (Area 1)
    # P_mod[1] = A1 % M (Area 2)
    # ...
    # P_mod[N-1] = (A1 + ... + A_{N-1}) % M (Area N)
    
    # Let S = total sum of A_i
    total_sum_mod = P_mod[N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0
    # Case 1: s < t
    # (P_mod[t-1] - P_mod[s-1]) % M == 0  => P_mod[t-1] == P_mod[s-1]
    # Case 2: s > t
    # (total_sum_mod - P_mod[s-1] + P_mod[t-1]) % M == 0 
    # => P_mod[s-1] - P_mod[t-1] == total_sum_mod % M
    
    # We only care about the first N prefix sums (indices 0 to N-1)
    current_P = P_mod[:N]
    counts = Counter(current_P)
    
    # For a fixed remainder r, there are counts[r] positions.
    # Pairs (s, t) with s < t and P_mod[s-1] == P_mod[t-1] == r:
    # There are counts[r] * (counts[r] - 1) // 2 such pairs.
    # However, the problem asks for the number of pairs (s, t).
    # Let's re-evaluate:
    # For every pair of indices i, j in {0, ..., N-1} with i < j:
    # If P_mod[i] == P_mod[j], then distance(i+1, j+1) is a multiple of M.
    # If (P_mod[i] - P_mod[j]) % M == total_sum_mod % M, then distance(j+1, i+1) is a multiple of M.
    
    # Let's use the property: 
    # Pair (s, t) is valid if:
    # 1. s < t and P_mod[t-1] ≡ P_mod[s-1] (mod M)
    # 2. s > t and P_mod[s-1] - P_mod[t-1] ≡ total_sum_mod (mod M)
    
    # Total valid pairs = Sum_{r=0 to M-1} [ (counts[r] * (counts[r]-1) // 2) 
    #                                         + (counts[r] * counts[(r - total_sum_mod) % M]) ]
    # Wait, the second term counts pairs (s, t) where s > t.
    # If total_sum_mod % M == 0, then the second term is counts[r] * counts[r].
    # But we must ensure s != t.
    # If total_sum_mod % M == 0, then s > t and P_mod[s-1] == P_mod[t-1] is the condition.
    # That would be counts[r] * (counts[r]-1) // 2.
    
    # Correct logic:
    # For every pair {i, j} with 0 <= i < j < N:
    # Check if (P_mod[j] - P_mod[i]) % M == 0  --> (s=i+1, t=j+1)
    # Check if (total_sum_mod - P_mod[j] + P_mod[i]) % M == 0 --> (s=j+1, t=i+1)
    
    # Let's use the Counter:
    # For each r in counts:
    # Ways to pick i < j such that P_mod[i] == P_mod[j] == r: counts[r] * (counts[r]-1) // 2
    # Ways to pick i < j such that (total_sum_mod - P_mod[j] + P_mod[i]) % M == 0:
    # This is P_mod[j] - P_mod[i] ≡ total_sum_mod (mod M)
    # Let r1 = P_mod[i] and r2 = P_mod[j]. We need r2 - r1 ≡ total_sum_mod (mod M).
    # This means r2 ≡ r1 + total_sum_mod (mod M).
    
    # If total_sum_mod % M == 0:
    # Both conditions are the same: P_mod[i] == P_mod[j].
    # Each such pair {i, j} gives two valid (s, t) pairs: (i+1, j+1) and (j+1, i+1).
    # Total = Sum(counts[r] * (counts[r]-1))
    
    # If total_sum_mod % M != 0:
    # Condition 1: P_mod[i] == P_mod[j]. Ways: Sum(counts[r] * (counts[r]-1) // 2)
    # Condition 2: P_mod[j] - P_mod[i] ≡ total_sum_mod (mod M).
    # Since i < j, we can't simply multiply counts. 
    # Actually, the condition s > t is just the "clockwise" distance from s to t.
    # The distance from s to t is:
    # If s < t: P_mod[t-1] - P_mod[s-1]
    # If s > t: (P_mod[N] - P_mod[s-1]) + P_mod[t-1]
    
    # Let x = P_mod[s-1] and y = P_mod[t-1].
    # If s < t: (y - x) % M == 0  => y ≡ x (mod M)
    # If s > t: (total_sum_mod - x + y) % M == 0 => x - y ≡ total_sum_mod (mod M)
    
    # Let's iterate over all pairs of remainders (r1, r2):
    # For a fixed pair of remainders r1, r2:
    # Number of s < t such that P_mod[s-1]=r1 and P_mod[t-1]=r2 is not easily found by Counter.
    # WAIT: The condition s < t is only to avoid double counting the same pair of points.
    # But (s, t) is an ordered pair.
    # For any two distinct indices i, j in {0, ..., N-1}:
    # If i < j:
    #   Check if (P_mod[j] - P_mod[i]) % M == 0
    #   Check if (total_sum_mod - P_mod[j] + P_mod[i]) % M == 0
    # If i > j:
    #   Check if (P_mod[i] - P_mod[j]) % M == 0 is NOT the case.
    #   The distance from s=i+1 to t=j+1 is (total_sum_mod - P_mod[i] + P_mod[j])
    
    # Let's simplify:
    # We want pairs (s, t) with s != t such that dist(s, t) % M == 0.
    # dist(s, t) = (P_mod[t-1] - P_mod[s-1]) % total_sum
    # In modulo M:
    # If s < t: (P_mod[t-1] - P_mod[s-1]) % M == 0
    # If s > t: (P_mod[N] - P_mod[s-1] + P_mod[t-1]) % M == 0
    
    # Let's use the remainders:
    # For every pair (s, t) with s < t:
    # Pair (s, t) is valid if P_mod[t-1] ≡ P_mod[s-1] (mod M)
    # Pair (t, s) is valid if P_mod[N] - P_mod[t-1] + P_mod[s-1] ≡ 0 (mod M)
    # <=> P_mod[t-1] - P_mod[s-1] ≡ P_mod[N] (mod M)
    
    # Let C = Counter(P_mod[:N])
    # For each r:
    #   Ways to pick s < t such that P_mod[s-1] == P_mod[t-1] == r:
    #   C[r] * (C[r] - 1) // 2
    # For each r:
    #   Ways to pick s < t such that P_mod[t-1] - P_mod[s-1] ≡ P_mod[N] (mod M):
    #   This is tricky because of the s < t constraint.
    #   Actually, if we pick any two indices i, j such that P_mod[j] - P_mod[i] ≡ P_mod[N] (mod M),
    #   then if i < j, the pair (t=j+1, s=i+1) is NOT necessarily the one.
    #   Let's re-read: "distance clockwise from s to t".
    #   If s < t, dist = P_mod[t-1] - P_mod[s-1]
    #   If s > t, dist = P_mod[N] - P_mod[s-1] + P_mod[t-1]
    
    # Let's use the property:
    # (s, t) is valid if:
    # 1. s < t AND P_mod[t-1] ≡ P_mod[s-1] (mod M)
    # 2. s > t AND P_mod[s-1] - P_mod[t-1] ≡ P_mod[N] (mod M)
    
    # For a fixed pair of indices {i, j} with i < j:
    # (s=i+1, t=j+1) is valid if P_mod[j] ≡ P_mod[i] (mod M)
    # (s=j+1, t=i+1) is valid if P_mod[j] - P_mod[i] ≡ P_mod[N] (mod M)
    
    # Total = Sum_{i < j} [ (1 if P_mod[j]==P_mod[i] else 0) + (1 if (P_mod[j]-P_mod[i])%M == P_mod[N]%M else 0) ]
    # Total = Sum_{r} (C[r]*(C[r]-1)//2) + Sum_{i < j} (1 if (P_mod[j]-P_mod[i])%M