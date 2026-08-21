The constraint to avoid explicit `for` and `while` loops forces a transition from imperative programming to a more functional style. While explicit loops are often more readable for complex state changes, using comprehensions and built-in higher-order functions can lead to concise, expression-oriented code that leverages Python's internal optimizations. For this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances and `collections.Counter` to efficiently count the occurrences of each remainder modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map input to integers
    # N: number of rest areas, M: the divisor
    # A: list of steps between area i and i+1
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))
    
    # Let S[i] be the distance from area 1 to area i+1.
    # S[0] = 0
    # S[1] = A[0]
    # S[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is (S[t-1] - S[s-1])
    # The distance from s to t (s > t) is (Total_Sum - S[s-1]) + S[t-1]
    
    # Calculate prefix sums modulo M
    # accumulate(A, lambda x, y: (x + y) % M, initial=0) gives [0, S1%M, S2%M, ..., SN%M]
    # Note: S[N] is the total sum of A.
    prefix_sums = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # The total sum of A modulo M
    total_sum_mod = prefix_sums[N]
    
    # We want (dist from s to t) % M == 0.
    # Case 1: s < t
    # (S[t-1] - S[s-1]) % M == 0  =>  S[t-1] % M == S[s-1] % M
    # Case 2: s > t
    # (Total_Sum - S[s-1] + S[t-1]) % M == 0  =>  S[s-1] % M == (Total_Sum + S[t-1]) % M
    
    # We only care about S[0]...S[N-1] for the starting/ending points
    # S[N] is the total sum, used for the wrap-around calculation.
    S_limited = prefix_sums[:N]
    counts = Counter(S_limited)
    
    # For a fixed s, we need t such that:
    # If s < t: S[t-1] % M == S[s-1] % M
    # If s > t: S[t-1] % M == (S[s-1] - Total_Sum) % M
    
    # Let's iterate through all possible remainders r in S_limited.
    # For each r, there are counts[r] positions.
    # Pairs (s, t) with s < t and S[s-1] == S[t-1] == r:
    # This is simply (counts[r] * (counts[r] - 1)) // 2
    # However, the problem asks for pairs (s, t). 
    # For a fixed s, we need t != s.
    # If s < t, we need S[t-1] % M == S[s-1] % M.
    # If s > t, we need S[t-1] % M == (S[s-1] - total_sum_mod) % M.
    
    # Total pairs = Sum for all r in counts:
    # counts[r] * (counts[r] - 1)  <-- This covers s < t and s > t where S[s-1] == S[t-1]
    # BUT, the condition for s > t is different.
    
    # Correct Logic:
    # For each s from 1 to N:
    # Let r = S[s-1] % M.
    # We need t such that:
    # 1. t > s and S[t-1] % M == r
    # 2. t < s and S[t-1] % M == (r - total_sum_mod) % M
    
    # Let's use the property:
    # Total = Sum_{r} (counts[r] * counts[(r - total_sum_mod) % M])
    # This counts pairs (s, t) such that S[t-1] % M == (S[s-1] - total_sum_mod) % M.
    # Wait, that's for s > t.
    
    # Let's re-evaluate:
    # For a fixed s, t is valid if:
    # (t > s AND S[t-1] % M == S[s-1] % M) OR (t < s AND S[t-1] % M == (S[s-1] - total_sum_mod) % M)
    
    # Let r1 = S[s-1] % M and r2 = S[t-1] % M.
    # Pair (s, t) is valid if:
    # (s < t AND r1 == r2) OR (s > t AND r2 == (r1 - total_sum_mod) % M)
    
    # Summing over all s, t:
    # Ans = Sum_{r} (counts[r] * (counts[r] - 1) / 2)  <-- for s < t
    #     + Sum_{r} (counts[r] * counts[(r - total_sum_mod) % M]) <-- for s > t
    # But we must be careful if (r - total_sum_mod) % M == r.
    # If total_sum_mod == 0, then (r - 0) % M == r.
    # Then for s > t, we need r2 == r1.
    # Total = Sum_{r} (counts[r] * (counts[r] - 1) / 2) + Sum_{r} (counts[r] * (counts[r] - 1) / 2)
    #       = Sum_{r} (counts[r] * (counts[r] - 1))
    
    # If total_sum_mod != 0:
    # For s < t: we need r1 == r2.
    # For s > t: we need r2 == (r1 - total_sum_mod) % M.
    # Since total_sum_mod != 0, r1 != (r1 - total_sum_mod) % M.
    # So the two sets of pairs are disjoint.
    # Ans = Sum_{r} [ counts[r] * (counts[r] - 1) / 2 ] + Sum_{r} [ counts[r] * counts[(r - total_sum_mod) % M] ]
    # Wait, the second term is for s > t. For a fixed r1, the number of t < s is not constant.
    
    # Let's use the indices.
    # Let indices[r] be the list of indices i where S[i] % M == r.
    # For a fixed s (index i), t (index j):
    # If i < j: valid if S[i] == S[j]
    # If i > j: valid if S[j] == (S[i] - total_sum_mod) % M
    
    # Let', target_r = (r - total_sum_mod) % M.
    # For each r:
    # Pairs (i, j) with i < j and S[i]=S[j]=r: counts[r] * (counts[r]-1) // 2
    # Pairs (i, j) with i > j and S[i]=r, S[j]=target_r: 
    # This is trickier. Let', target_r = (r - total_sum_mod) % M.
    # We need to count pairs (j, i) such that j < i, S[j] = target_r, S[i] = r.
    
    # Let's use a different approach:
    # For every pair of remainders (r_j, r_i) where j < i:
    # The pair (s, t) = (j+1, i+1) is valid if r_j == r_i.
    # The pair (s, t) = (i+1, j+1) is valid if r_j == (r_i - total_sum_mod) % M.
    
    # Total = Sum_{r} (counts[r] * (counts[r] - 1) // 2)  <-- for s < t
    #       + Sum_{r} (count of j < i such that S[j] = (S[i] - total_sum_mod) % M)
    
    # To calculate the second term without loops:
    # We need Sum_{i=0}^{N-1} (count of j < i such that S[j] == (S[i] - total_sum_mod) % M)
    # This can be done by iterating through S and keeping track of counts.
    # But we can't use loops.
    # We can use a custom reduce or a clever combination of map/sum.
    # Actually, we can use the fact that:
    # Sum_{j < i} [S[j] == target_r] = 
    # (counts[target_r] * counts[r]) - (count of j > i such that S[j] == target_r)
    
    # Let' same_r = total_sum_mod == 0
    # If same_r:
    #   Ans = Sum_{r} (counts[r] * (counts[r] - 1))
    # If not same_r:
    #   Part 1 (s < t): Sum_{r} (counts[r] * (counts[r] - 1) // 2)
    #   Part 2 (s > t): Sum_{i=0}^{N-1} (count of j < i where S[j] == (S[i] - total_sum_mod) % M)
    
    # To compute Part 2 without loops:
    # Let r_i = S[i]. We want Sum_{i} Sum_{j < i} [r_j == (r_i - total_sum_mod) % M].
    # Let target(r) = (r - total_sum_mod) % M.
    # We want Sum_{i} (count of j < i with r_j == target(r_i)).
    # This is a convolution-like sum.
    # Let's use the property: 
    # Sum_{j < i} [r_j == target_r] = 
    # (Overall count of target_r) - (count of j >= i with r_j == target_r).
    
    # Alternatively, we can use a list to store the "first occurrence" or "last occurrence" 
    # but that's not quite right.
    
    # Let's use the "total count" logic:
    # For a fixed r, let target_r = (r - total_sum_mod) % M.
    # We want to count pairs (j, i) such that j < i, S[j] = target_r, S[i] = r.
    # If target_r == r, this is just counts[r] * (counts[r] - 1) // 2.
    # If target_r != r, we can't simply use counts[r] * counts[target_r].
    # We need the relative order.
    
    # Wait, we can use a list comprehension to build a list of "how many target_r seen so far".
    # But we can't update a counter in a list comprehension.
    # Actually, we can use a helper function with a mutable object or a closure, 
    # but that's basically a loop.
    
    # Let' same_r = (total_sum_mod == 0)
    # If same_r:
    #     ans = sum((counts[r] * (counts[r] - 1)) for r in counts)
    # Else:
    #     # Part 1: s < t (r_s == r_t)
    #     p1 = sum((counts[r] * (counts[r] - 1) // 2) for r in counts)
    #     # Part 2: s > t (r_t == (r_