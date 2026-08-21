The constraint to avoid explicit loops forces a transition from imperative programming to a functional paradigm. This encourages the use of high-level abstractions and vectorized-style operations, which often lead to more concise code and can be more efficient when using libraries like NumPy (though not allowed here). To solve this problem without loops, I will use `itertools.accumulate` to calculate prefix sums of the distances, and `collections.Counter` combined with generator expressions to count the occurrences of each remainder modulo $M$.

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
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A[0]
    # P[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Calculate prefix sums
    # P will have N+1 elements: [0, A1, A1+A2, ..., sum(A)]
    P = list(accumulate(A, initial=0))
    total_sum = P[N]
    
    # We are interested in P[i] % M for i in 0...N-1
    # Let R[i] = P[i] % M
    R = [p % M for p in P[:-1]]
    counts = Counter(R)
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => R[t-1] == R[s-1]
    # For each remainder r, if it appears k times, there are k*(k-1)//2 pairs.
    ans_st = sum(k * (k - 1) // 2 for k in counts.values())
    
    # For a pair (s, t) with s > t:
    # Distance is (total_sum - P[s-1] + P[t-1]) % M == 0
    # => P[t-1] - P[s-1] == -total_sum (mod M)
    # => R[t-1] - R[s-1] == (M - (total_sum % M)) % M
    # Let target = (M - (total_sum % M)) % M
    # We need to find pairs (s, t) such that R[t-1] - R[s-1] == target (mod M)
    # where t < s.
    # This is equivalent to counting pairs (R[i], R[j]) such that 
    # R[i] - R[j] == target (mod M) for all i, j in 0...N-1, i != j.
    # Wait, the s < t and s > t logic can be simplified:
    # For any two distinct indices i, j from {0, ..., N-1}:
    # If i < j, we check (P[j] - P[i]) % M == 0
    # If i > j, we check (total_sum - P[i] + P[j]) % M == 0
    
    # Let's redefine:
    # For every pair {i, j} with 0 <= i < j < N:
    # Pair (s=i+1, t=j+1) is valid if (P[j] - P[i]) % M == 0
    # Pair (s=j+1, t=i+1) is valid if (total_sum - P[j] + P[i]) % M == 0
    
    # Part 1: (P[j] - P[i]) % M == 0  => R[j] == R[i]
    # This is already calculated as ans_st.
    
    # Part 2: (total_sum - P[j] + P[i]) % M == 0 => R[j] - R[i] == total_sum % M
    # Let T = total_sum % M.
    # We need to count pairs (i, j) with i < j such that R[j] - R[i] == T (mod M).
    # This is slightly different from just counting frequencies because of the i < j constraint.
    # HOWEVER, the problem asks for the number of pairs (s, t).
    # For any two distinct rest areas s and t, there is exactly one clockwise path from s to t.
    # Let's use the property:
    # Dist(s, t) = (P[t-1] - P[s-1]) mod total_sum
    # Dist(s, t) % M == 0  <=>  (P[t-1] - P[s-1]) % M == 0 if s < t
    # Dist(s, t) % M == 0  <=>  (total_sum + P[t-1] - P[s-1]) % M == 0 if s > t
    
    # Let R[i] = P[i] % M.
    # If s < t: R[t-1] - R[s-1] \equiv 0 (mod M)
    # If s > t: R[t-1] - R[s-1] \equiv -total_sum (mod M)
    
    # Let T = total_sum % M.
    # We want to count pairs (i, j) such that:
    # 1. 0 <= i < j < N and R[j] - R[i] \equiv 0 (mod M)
    # 2. 0 <= j < i < N and R[j] - R[i] \equiv -T (mod M)
    
    # Notice that for any two distinct indices i, j:
    # If R[i] == R[j], then one of the directions (either i->j or j->i) 
    # will have a distance divisible by M IF T == 0.
    # If T == 0, then (P[j]-P[i]) % M == 0 AND (total_sum - P[j] + P[i]) % M == 0.
    # If T != 0, then only one of the two directions can be divisible by M.
    
    # Let's use a different approach:
    # For each i in 0...N-1, we want to count j != i such that:
    # If i < j: R[j] == R[i]
    # If i > j: R[j] == (R[i] - T) % M
    
    # Total count = sum_{i=0 to N-1} [ count(j > i where R[j] == R[i]) + count(j < i where R[j] == (R[i] - T) % M) ]
    
    # This can be solved by iterating through the list and keeping track of counts.
    # Since we can't use loops, we can use a combination of:
    # 1. Total pairs (i, j) where R[j] == R[i] (which is sum k*(k-1))
    # 2. But we need to distinguish between i < j and i > j.
    
    # Let's reconsider:
    # For a fixed pair of indices {i, j} with i < j:
    # Direction i -> j is valid if R[j] - R[i] == 0 (mod M)
    # Direction j -> i is valid if R[i] - R[j] == -T (mod M) => R[j] - R[i] == T (mod M)
    
    # So for every pair {i, j} with i < j:
    # - It contributes 1 to the answer if R[i] == R[j]
    # - It contributes 1 to the answer if (R[j] - R[i]) % M == T
    
    # Special case: If T == 0, then R[i] == R[j] and (R[j] - R[i]) % M == T are the same condition.
    # But the problem says s != t, and we are checking clockwise.
    # If T == 0, and R[i] == R[j], then both s=i+1, t=j+1 AND s=j+1, t=i+1 are valid.
    
    # General case:
    # Count pairs (i, j) with i < j such that R[j] - R[i] == 0 (mod M)
    # PLUS count pairs (i, j) with i < j such that R[j] - R[i] == T (mod M)
    
    # Let's use the frequency map 'counts'.
    # The number of pairs {i, j} with R[i] == R[j] is sum(k*(k-1)//2).
    # The number of pairs {i, j} with R[j] - R[i] == T (mod M) is:
    # If T == 0: sum(k*(k-1)//2)
    # If T != 0: we need to count pairs (i, j) with i < j such that R[j] - R[i] == T.
    # This depends on the order. We can't just use the Counter.
    
    # Wait, the condition "i < j" for R[j] - R[i] == T is only for the "backward" direction.
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (total_sum - P[s-1] + P[t-1]) % M == 0
    
    # Let R[i] = P[i] % M.
    # Condition 1: R[t-1] == R[s-1] (with s < t)
    # Condition 2: R[t-1] - R[s-1] == -total_sum % M (with s > t)
    
    # Let T = total_sum % M.
    # Condition 2 is R[t-1] - R[s-1] == (M - T) % M (with s > t).
    # Let T' = (M - T) % M.
    # We want to count pairs (i, j) such that:
    # (i < j and R[j] == R[i]) OR (i > j and R[j] - R[i] == T')
    
    # Let's use the fact that:
    # Total pairs (i, j) with i != j such that R[j] - R[i] == T' (mod M)
    # is sum(counts[r] * counts[(r + T') % M]) for all r.
    # If T' == 0, this is sum(k * (k-1)).
    # If T' != 0, this is sum(counts[r] * counts[(r + T') % M]).
    
    # Is there a symmetry?
    # For any pair {i, j} with i < j:
    # Direction i -> j is valid if R[j] - R[i] == 0 (mod M)
    # Direction j -> i is valid if R[i] - R[j] == -T (mod M) => R[j] - R[i] == T (mod M)
    
    # Total = Count({i, j} : i < j and R[i] == R[j]) + Count({i, j} : i < j and R[j] - R[i] == T (mod M))
    # If T == 0, the two conditions are identical, but they refer to different pairs (s, t).
    # Specifically, if R[i] == R[j], then both (s=i+1, t=j+1) and (s=j+1, t=i+1) are valid.
    # So if T == 0, answer is sum(k * (k-1)).
    # If T != 0, we need to count pairs (i, j) with i < j such that R[j] - R[i] == 0 OR R[j] - R[i] == T.
    # Note that R[j] - R[i] == 0 and R[j] - R[i] == T cannot both be true if T != 0.
    