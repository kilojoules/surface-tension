The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state updates. This approach leverages Python's highly optimized internal C-loops via built-ins and comprehensions, which can often be more concise and efficient for data processing tasks. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions to handle the modulo operations, and `collections.Counter` to efficiently count occurrences of each remainder.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = list(map(int, sys.stdin.read().split()))
    
    # N: number of rest areas, M: the divisor
    # A: list of distances between rest areas
    N, M = input_data[0], input_data[1]
    A = input_data[2:]
    
    # Let P[i] be the clockwise distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A[0]
    # P[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Calculate prefix sums
    # P will have N elements: [0, A1, A1+A2, ..., A1+...+A_{N-1}]
    # We use accumulate to get the prefix sums of A, then prepend 0.
    # Note: we only need the first N-1 prefix sums for the P array.
    P = list(accumulate(A[:-1], initial=0))
    total_sum = sum(A)
    
    # We want (dist from s to t) % M == 0.
    # Let R[i] = P[i] % M.
    # For s < t: (R[t-1] - R[s-1]) % M == 0  => R[t-1] == R[s-1]
    # For s > t: (total_sum - R[s-1] + R[t-1]) % M == 0 => R[s-1] - R[t-1] == total_sum % M
    
    R = [p % M for p in P]
    counts = Counter(R)
    
    # For s < t, the number of pairs is sum(count * (count - 1) // 2) for each unique remainder.
    # This is equivalent to counting pairs with the same remainder.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For s > t, we need R[s-1] - R[t-1] ≡ total_sum (mod M).
    # Let T = total_sum % M.
    # We need R[s-1] ≡ R[t-1] + T (mod M).
    # For each remainder r, the number of pairs is counts[r] * counts[(r + T) % M].
    # However, we must exclude the case where s=t, but the problem says s != t.
    # Since s > t, s and t are distinct.
    # Special case: If T == 0, then R[s-1] == R[t-1]. 
    # The number of pairs (s, t) with s > t and R[s-1] == R[t-1] is the same as s < t.
    
    T = total_sum % M
    
    # Use a generator expression to calculate the sum for s > t
    # We iterate over the unique remainders present in R.
    ans_s_gt_t = sum(counts[r] * counts.get((r + T) % M, 0) for r in counts)
    
    # If T == 0, the logic above counts pairs where R[s-1] == R[t-1].
    # But we must ensure s != t. The loop `for r in counts` includes the case 
    # where we multiply counts[r] by itself. 
    # Actually, the condition is s > t. 
    # For a fixed remainder r, there are counts[r] indices.
    # The number of pairs (s, t) with s > t and R[s-1] == R[t-1] is counts[r] * (counts[r] - 1) // 2.
    # But the logic `counts[r] * counts[(r + T) % M]` when T=0 gives counts[r]^2.
    # Let's refine:
    
    # If T == 0:
    # Pairs (s, t) with s < t and R[s-1] == R[t-1]: sum(c*(c-1)//2)
    # Pairs (s, t) with s > t and R[s-1] == R[t-1]: sum(c*(c-1)//2)
    # Total = sum(c*(c-1))
    
    # If T != 0:
    # Pairs (s, t) with s < t and R[s-1] == R[t-1]: sum(c*(c-1)//2)
    # Pairs (s, t) with s > t and R[s-1] - R[t-1] == T: sum(counts[r] * counts[(r-T)%M])
    
    # Let's use a more robust calculation for s > t:
    # For every t < s, we check if (total_sum - P[s-1] + P[t-1]) % M == 0
    # Which is P[s-1] - P[t-1] ≡ total_sum (mod M)
    # Let R[i] = P[i] % M.
    # R[s-1] - R[t-1] ≡ T (mod M)  => R[t-1] ≡ R[s-1] - T (mod M)
    
    # Corrected logic:
    # 1. Pairs (s, t) with s < t: R[s-1] == R[t-1]
    # 2. Pairs (s, t) with s > t: R[t-1] == (R[s-1] - T) % M
    
    # For s < t:
    res1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For s > t:
    # We need to count pairs (i, j) such that 0 <= j < i < N and R[j] == (R[i] - T) % M.
    # This is harder without a loop because it depends on the order.
    # WAIT: The condition is simply that the clockwise distance from s to t is a multiple of M.
    # Clockwise distance from s to t:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: (Total - P[s-1]) + P[t-1]
    
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # 1. s < t AND (P[t-1] - P[s-1]) % M == 0
    # 2. s > t AND (Total + P[t-1] - P[s-1]) % M == 0
    
    # Let R[i] = P[i] % M.
    # 1. s < t: R[t-1] == R[s-1]
    # 2. s > t: R[t-1] == (R[s-1] - Total) % M
    
    # Let T = Total % M.
    # For a fixed index i (representing rest area i+1):
    # It can be the 't' in s < t if there's some j < i with R[j] == R[i].
    # It can be the 's' in s > t if there's some j < i with R[j] == (R[i] - T) % M.
    
    # Total pairs = sum_{i=0 to N-1} [ (count of R[j]==R[i] for j < i) + (count of R[j]==(R[i]-T)%M for j < i) ]
    # This can be solved by iterating and keeping track of counts, but loops are forbidden.
    # However, we can use the property:
    # Total pairs = (Pairs with R[s-1] == R[t-1]) + (Pairs with R[t-1] == (R[s-1] - T) % M)
    # But the second condition is for s > t.
    
    # Let's use the property:
    # Total = sum_{i < j} [R[i] == R[j]] + sum_{i > j} [R[j] == (R[i] - T) % M]
    # The first term is sum(c * (c-1) // 2)
    # The second term: for each i, we need count of j < i such that R[j] == (R[i] - T) % M.
    # This is still order-dependent. Let's use the fact that:
    # sum_{i > j} [R[j] == (R[i] - T) % M] + sum_{i < j} [R[j] == (R[i] - T) % M] 
    # = sum_{r} (counts[r] * counts[(r - T) % M]) 
    # (Subtracting the case i=j if (R[i] - T) % M == R[i], which happens if T == 0)
    
    # Let's use the symmetry.
    # Let f(r1, r2) be the number of pairs (i, j) with i < j such that R[i]=r1 and R[j]=r2.
    # We want: sum_{r} f(r, r) + sum_{r} f((r-T)%M, r)
    # Note that f(r1, r2) + f(r2, r1) = counts[r1] * counts[r2] (for r1 != r2)
    # And f(r, r) = counts[r] * (counts[r] - 1) // 2
    
    # If T == 0:
    # Result = sum f(r, r) + sum f(r, r) = sum(counts[r] * (counts[r] - 1))
    
    # If T != 0:
    # Result = sum f(r, r) + sum f((r-T)%M, r)
    # This is still tricky because f((r-T)%M, r) depends on the order of indices.
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # For any s != t, there is only one way to go clockwise.
    # The distance is (P[t-1] - P[s-1]) % Total.
    # We want (P[t-1] - P[s-1]) % Total to be a multiple of M.
    # This is equivalent to (P[t-1] - P[s-1]) % M == 0, provided Total is a multiple of M?
    # No. The distance is:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: Total - (P[s-1] - P[t-1])
    # Both are equivalent to (P[t-1] - P[s-1]) mod Total.
    # We want (P[t-1] - P[s-1]) mod Total ≡ 0 (mod M).
    # This is simply P[t-1] - P[s-1] ≡ 0 (mod M) IF Total ≡ 0 (mod M).
    # If Total is not 0 mod M, then for s > t, the condition is:
    # Total + P[t-1] - P[s-1] ≡ 0 (mod M)  => P[s-1] - P[t-1] ≡ Total (mod M).
    
    # Let's use the property:
    # For any pair {i, j} with i < j:
    # One is s=i+1, t=j+1 (dist = P[j] - P[i])
    # One is s=j+1, t=i+1 (dist = Total - (P[j] - P[i]))
    # We want to count how many of these are 0 mod M.
    