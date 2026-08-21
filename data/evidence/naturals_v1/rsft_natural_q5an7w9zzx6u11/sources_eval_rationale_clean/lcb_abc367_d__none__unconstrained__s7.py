The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing what to calculate rather than how to iterate. In Python, this encourages the use of high-level built-ins and `itertools`, which are often more concise and can be more performant for specific data transformations. To solve this problem, I will use `itertools.accumulate` to compute prefix sums of the distances, list comprehensions for data transformation, and `collections.Counter` to efficiently count occurrences of remainders modulo $M$.

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
    # P[1] = A[0]
    # P[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Compute prefix sums: P[0], P[1], ..., P[N-1]
    # accumulate(A, initial=0) provides the distance from area 1 to area 1, 2, ..., N
    # We only need the first N prefix sums.
    P = list(accumulate(A, initial=0))[:N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # Let R[i] = P[i] % M.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => R[t-1] == R[s-1]
    # For s > t: (Total_Sum - P[s-1] + P[t-1]) % M == 0 => (R[t-1] - R[s-1]) % M == (-Total_Sum) % M
    
    # Total sum of all A_i
    total_sum_mod = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(map(lambda x: x % M, P))
    
    # For a fixed remainder r, there are counts[r] positions.
    # The number of pairs (s, t) with s < t and R[s-1] == R[t-1] is:
    # sum(c * (c - 1) // 2 for c in counts.values())
    # However, the problem asks for all pairs (s, t).
    
    # Let's analyze the condition:
    # Distance from s to t is (P[t-1] - P[s-1]) mod M if s < t
    # Distance from s to t is (Total_Sum + P[t-1] - P[s-1]) mod M if s > t
    
    # Let R_i = P[i] % M.
    # We want (R_{t-1} - R_{s-1}) % M == 0 for s < t
    # We want (Total_Sum + R_{t-1} - R_{s-1}) % M == 0 for s > t
    
    # This is equivalent to:
    # s < t: R_{s-1} == R_{t-1}
    # s > t: R_{s-1} == (R_{t-1} + Total_Sum) % M
    
    # Let's calculate the contribution of each remainder r:
    # For each r, there are counts[r] indices.
    # Pairs (s, t) with s < t and R_{s-1} = R_{t-1} = r: counts[r] * (counts[r] - 1) // 2
    # Pairs (s, t) with s > t and R_{s-1} = (r + Total_Sum) % M and R_{t-1} = r:
    # This is slightly trickier because the s > t condition depends on the indices.
    
    # Alternative approach:
    # For every pair (i, j) with i != j:
    # If i < j, distance is (P[j] - P[i]) % M
    # If i > j, distance is (Total_Sum + P[j] - P[i]) % M
    
    # Total pairs = sum_{i < j} [P[j]-P[i] == 0 mod M] + sum_{i > j} [Total_Sum + P[j]-P[i] == 0 mod M]
    # = sum_{r} (counts[r] * (counts[r]-1) // 2) + sum_{i > j} [P[i]-P[j] == Total_Sum mod M]
    
    # Let's evaluate the second term: sum_{i > j} [P[i] - P[j] == Total_Sum mod M]
    # This is sum_{i=1 to N-1} (count of j < i such that P[j] == (P[i] - Total_Sum) mod M)
    
    # To avoid loops, we can use a trick with the total counts:
    # The total number of pairs (i, j) with i != j such that (P[i] - P[j]) % M == Total_Sum % M
    # is sum(counts[r] * counts[(r - total_sum_mod) % M]) 
    # But we must subtract cases where i == j (which only happens if Total_Sum % M == 0)
    # and then handle the i < j vs i > j split.
    
    # Actually, the simplest way:
    # For every pair i, j (i != j):
    # If i < j, we need P[j] - P[i] = 0 mod M
    # If i > j, we need P[j] - P[i] = -Total_Sum mod M  => P[i] - P[j] = Total_Sum mod M
    
    # Let's use the property:
    # Total = sum_{i < j} [R_j == R_i] + sum_{i > j} [R_i - R_j == Total_Sum mod M]
    
    # Let's compute the first term:
    term1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For the second term, we need to count pairs (i, j) with i > j and R_i - R_j == Total_Sum mod M.
    # This is equivalent to counting pairs (i, j) with i > j and R_j == (R_i - Total_Sum) mod M.
    # We can compute this by iterating through the list and keeping track of counts of R_j seen so far.
    # Since we can't use loops, we can use a combination of a custom function and a reducer or 
    # just realize that:
    # sum_{i > j} [R_i - R_j == T] = (sum_{all i, j} [R_i - R_j == T] - sum_{i=j} [R_i - R_j == T]) / 2 
    # ONLY IF T == 0. If T != 0, the i < j and i > j cases are different.
    
    # Wait, if T == 0, then R_i == R_j. The number of pairs i > j is the same as i < j.
    # If T != 0, we can't simply divide by 2.
    
    # Let's use a different approach for the second term:
    # We want to count pairs (i, j) such that i > j and R_i - R_j = T (mod M).
    # This is sum_{i=0}^{N-1} (count of j < i such that R_j = (R_i - T) mod M).
    
    # To do this without loops, we can use a technique with a running total.
    # However, since we can't use loops, we can't easily maintain a state.
    # But we can use the fact that:
    # sum_{i > j} [R_i - R_j = T] + sum_{i < j} [R_i - R_j = T] = sum_{R_i - R_j = T, i != j} 1
    # And sum_{i < j} [R_i - R_j = T] is the same as sum_{j > i} [R_j - R_i = -T]
    
    # Let T = total_sum_mod.
    # Term 1: sum_{i < j} [R_j - R_i = 0]
    # Term 2: sum_{i > j} [R_i - R_j = T]
    
    # If T == 0:
    # Term 1 = sum(c*(c-1)//2), Term 2 = sum(c*(c-1)//2). Total = sum(c*(c-1))
    
    # If T != 0:
    # We need sum_{i < j} [R_j - R_i = 0] + sum_{i > j} [R_i - R_j = T]
    # Note that sum_{i > j} [R_i - R_j = T] is the number of pairs (i, j) with i > j and R_j = R_i - T.
    # This is not easily solvable with just Counter without a loop.
    # UNLESS we realize that the total distance from s to t is just (P[t-1] - P[s-1]) mod Total_Sum.
    # No, that's not right.
    
    # Let's use the property:
    # The distance from s to t is (P[t-1] - P[s-1]) % Total_Lake_Length.
    # We want (P[t-1] - P[s-1]) % M == 0.
    # This is equivalent to P[t-1] % M == P[s-1] % M.
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # From s to t (s < t), the distance is P[t-1] - P[s-1].
    # From s to t (s > t), the distance is (Total_Sum - P[s-1]) + P[t-1].
    
    # Let R_i = P[i] % M.
    # s < t: (R_{t-1} - R_{s-1}) % M == 0  => R_{t-1} == R_{s-1}
    # s > t: (Total_Sum + R_{t-1} - R_{s-1}) % M == 0 => R_{s-1} - R_{t-1} == Total_Sum % M
    
    # Let T = Total_Sum % M.
    # We want to count pairs (i, j) such that:
    # 1. i < j and R_i == R_j
    # 2. i > j and R_i - R_j == T (mod M)
    
    # Let's use the identity:
    # sum_{i < j} [R_i == R_j] = (sum_{i, j} [R_i == R_j] - N) / 2
    # sum_{i > j} [R_i - R_j == T] = sum_{i, j} [R_i - R_j == T] - sum_{i < j} [R_i - R_j == T] - sum_{i=j} [R_i - R_j == T]
    
    # This is getting complex. Let's simplify.
    # For a fixed pair of values (r1, r2) such that r1, r2 in [0, M-1]:
    # Let c1 = counts[r1], c2 = counts[r2].
    # If r1 == r2:
    #   Pairs (i, j) with i < j and R_i=r1, R_j=r1: c1*(c1-1)//2
    #   Pairs (i, j) with i > j and R_i=r1, R_j=r1: c1*(c1-1)//2 (if T == 0)
    # If r1 != r2:
    #   Pairs (i, j) with i