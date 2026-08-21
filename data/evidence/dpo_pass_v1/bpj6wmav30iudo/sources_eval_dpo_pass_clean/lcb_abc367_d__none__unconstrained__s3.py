The constraint to avoid explicit `for` and `while` loops forces a shift from imperative programming to a functional style. While explicit loops are often more readable for complex state changes, using comprehensions and built-in higher-order functions can lead to concise, expression-oriented code that leverages Python's internal optimizations. For this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances and `collections.Counter` to efficiently count the occurrences of each remainder modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use a trick to unpack the iterator
    # We can't use a for loop, but we can use a list comprehension 
    # or map to consume the input.
    
    # To handle the input without a loop, we convert the map to a list
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Let S[i] be the distance from rest area 1 to rest area i+1.
    # S[0] = 0
    # S[1] = A[0]
    # S[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is S[t-1] - S[s-1].
    # The distance from s to t (s > t) is (Total_Sum - S[s-1]) + S[t-1].
    
    # Calculate prefix sums modulo M
    # accumulate creates the sequence S[1], S[2]... S[N]
    # We prepend 0 to represent S[0]
    S = list(accumulate(A, lambda x, y: (x + y) % M))
    S = [0] + S
    
    # Total sum of A modulo M is the last element of the accumulate result
    # Note: S has N+1 elements. S[N] is the total sum.
    total_sum_mod = S[N]
    
    # We want (S[t-1] - S[s-1]) % M == 0 for s < t
    # and (total_sum_mod - S[s-1] + S[t-1]) % M == 0 for s > t.
    
    # Let's analyze the condition:
    # For a fixed s, we need t such that:
    # If t > s: S[t-1] ≡ S[s-1] (mod M)
    # If t < s: S[t-1] ≡ S[s-1] - total_sum_mod (mod M)
    
    # Let's count occurrences of each remainder in S[0...N-1]
    # S[N] is the total sum, but the rest areas are 1...N.
    # The prefix sums for the start/end points are S[0], S[1], ..., S[N-1].
    counts = Counter(S[:N])
    
    # For each s, we look for t.
    # If total_sum_mod == 0:
    #   The condition is always S[t-1] ≡ S[s-1] (mod M) regardless of whether t > s or t < s.
    #   For a specific remainder r, there are counts[r] positions.
    #   Each position can pair with any other position with the same remainder.
    #   Number of pairs = sum(counts[r] * (counts[r] - 1))
    # If total_sum_mod != 0:
    #   For a fixed s, we need t such that:
    #   1. t > s and S[t-1] ≡ S[s-1] (mod M)
    #   2. t < s and S[t-1] ≡ S[s-1] - total_sum_mod (mod M)
    #   This is equivalent to: for each s, count t's that satisfy the modular arithmetic.
    #   Actually, for any pair (s, t) with s != t, they form a valid pair if:
    #   (S[t-1] - S[s-1]) % M == 0 AND s < t
    #   OR (S[t-1] - S[s-1] + total_sum_mod) % M == 0 AND s > t
    
    # Let's simplify:
    # A pair (s, t) with s < t is valid if S[t-1] ≡ S[s-1] (mod M).
    # A pair (s, t) with s > t is valid if S[t-1] ≡ S[s-1] - total_sum_mod (mod M).
    
    # Let r1 = S[s-1] and r2 = S[t-1].
    # If s < t: r2 ≡ r1 (mod M)
    # If s > t: r2 ≡ r1 - total_sum_mod (mod M)
    
    # Total count = Sum_{r} (counts[r] * (counts[r] - 1)) if total_sum_mod == 0
    # If total_sum_mod != 0:
    # For each r, the number of s < t with S[s-1]=r and S[t-1]=r is counts[r]*(counts[r]-1)//2
    # For each r, the number of s > t with S[s-1]=r and S[t-1]=r-total_sum_mod is counts[r]*counts[(r-total_sum_mod)%M]
    # Wait, the second part is simpler:
    # For any two distinct indices i, j in {0...N-1}:
    # If i < j, they form (s, t) if S[j] ≡ S[i] (mod M)
    # If i > j, they form (s, t) if S[j] , S[i] satisfy the other condition.
    
    # Let's use the property:
    # Pair (s, t) is valid if:
    # 1. s < t and S[t-1] - S[s-1] ≡ 0 (mod M)
    # 2. s > t and S[t-1] - S[s-1] + total_sum_mod ≡ 0 (mod, M)
    
    # Let's calculate:
    # Part 1: sum(C(counts[r], 2)) for all r
    # Part 2: sum(counts[r] * counts[(r - total_sum_mod) % M]) for all r
    # But we must be careful not to double count or include s=t.
    
    # If total_sum_mod == 0:
    # Part 1: s < t, S[t-1] == S[s-1]
    # Part 2: s > t, S[t-1] == S[s-1]
    # Total = sum(counts[r] * (counts[r] - 1))
    
    # If total_sum_mod != 0:
    # Part 1: s < t, S[t-1] == S[s-1]
    # Part 2: s > t, S[t-1] == S[s-1] - total_sum_mod
    # Total = sum(counts[r] * (counts[r] - 1) // 2) + sum(counts[r] * counts[(r - total_sum_mod) % M])
    # Wait, the second sum is over all r. For a fixed r, we count how many t < s 
    # such that S[t-1] = (S[s-1] - total_sum_mod) % M.
    # This is simply counts[r] * counts[(r - total_sum_mod) % M].
    
    # Correct Logic:
    # Let, for each r in 0...M-1, uma be the number of indices i in 0...N-1 such that S[i] = r.
    # Pairs (s, t) with s < t: S[t-1] ≡ S[s-1] (mod M). 
    # For each r, there are counts[r] choose 2 such pairs.
    # Pairs (s, t) with s > t: S[t-1] ≡ S[s-1] - total_sum_mod (mod M).
    # For each r, there are counts[r] * counts[(r - total_sum_mod) % M] such pairs.
    # BUT, we must ensure t < s. 
    # Let' same_rem_pairs = sum(counts[r] * (counts[r] - 1) // 2)
    # Let diff_rem_pairs = sum(counts[r] * counts[(r - total_sum_mod) % M])
    # If total_sum_mod == 0, the conditions are the same, but s < t and s > t are disjoint.
    # result = same_rem_pairs * 2
    # If total_sum_mod != 0, the conditions are different.
    # For a fixed pair of indices {i, j} with i < j:
    # They can form (s, t) = (i+1, j+1) if S[j] ≡ S[i] (mod M)
    # They can form (s, t) = (j+1, i+1) if S[i] ≡ S[j] - total_sum_mod (mod M)
    # These two conditions are:
    # 1. S[j] - S[i] ≡ 0 (mod M)
    # 2. S[j] - S[i] ≡ total_sum_mod (mod M)
    
    # Total = sum(counts[r] * (counts[r] - 1) // 2) + sum(counts[r] * counts[(r - total_sum_mod) % M])
    # Wait, the second term: for each r, we count pairs (s, t) where s > t and S[t-1] = (S[s-1] - total_sum_mod) % M.
    # This is exactly sum_{r} (counts[r] * counts[(r - total_sum_mod) % M]).
    # However, if total_sum_mod == 0, this would be sum(counts[r]^2), which is wrong.
    
    # Let',s be the set of indices.
    # Ans = sum_{i < j} [S[j] == S[i]] + sum_{i < j} [S[i] == S[j] - total_sum_mod]
    # Ans = sum_{r} (counts[r] * (counts[r]-1)//2) + sum_{r} (counts[r] * counts[(r + total_sum_mod)%M])
    # But the second sum is over i < j. This is tricky.
    # Let's use: sum_{i < j} [S[i] == S[j] - total_sum_mod]
    # Let's iterate through the array S and maintain counts of elements seen so far.
    # For each S[j], we add:
    # 1. count of S[i] == S[j] for i < j
    # 2. count of S[i] == S[j] - total_sum_mod for i < j
    
    # Since we can't use loops, we use a custom reduce or a list comprehension with a dictionary.
    # But Python's dictionary update in a comprehension is a hack.
    # Let's use the mathematical property:
    # sum_{i < j} [S[i] == S[j] - total_sum_mod] 
    # If total_sum_mod == 0, this is the same as the first sum.
    # If total_sum_mod != 0, then S[i] and S[j] must be different.
    # For any two indices i, j with S[i] = r1 and S[j] = r2:
    # If r1 == r2, they contribute 1 to the total (as s < t).
    # If r2 - r1 ≡ total_sum_mod (mod M), they contribute 1 to the total (as s > t).
    
    # Let' same_rem = sum(counts[r] * (counts[r] - 1) // 2 for r in counts)
    # Let' diff_rem = sum(counts[r] * counts[(r - total_sum_mod) % M] for r in counts)
    # If total_sum_mod == 0:
    #   Ans = same_rem * 2