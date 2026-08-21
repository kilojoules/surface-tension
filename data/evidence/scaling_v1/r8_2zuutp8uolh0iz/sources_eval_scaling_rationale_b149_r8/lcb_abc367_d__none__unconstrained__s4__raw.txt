The constraint to avoid explicit loops forces a shift toward a functional programming style, emphasizing data transformations over stateful iteration. In Python, this encourages the use of built-in higher-order functions and comprehensions, which are often more concise and can be more performant for specific data processing tasks. To solve this problem, I will use `itertools.accumulate` to compute prefix sums of the distances, list comprehensions to handle the modulo operations, and `collections.Counter` to efficiently count occurrences of remainders, allowing for the calculation of valid pairs using a generator expression and `sum()`.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    # and a list comprehension to gather the remaining A_i
    data_iter = iter(input_data)
    N = next(data_iter)
    M = next(data_iter)
    A = [next(data_iter) for _ in range(N)]

    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P_t - P_s)
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t)
    
    # Compute prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + A) gives 0, A_1, A_1+A_2, ...
    # We only need the first N prefix sums.
    P = list(accumulate([0] + A))[:N]
    total_sum = sum(A)
    
    # We want (dist from s to t) % M == 0
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # Case 2: s > t
    # (total_sum - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == total_sum % M
    
    # Calculate remainders for all P_i
    rems = [p % M for p in P]
    counts = Counter(rems)
    
    # For Case 1 (s < t):
    # For each remainder r, if there are 'c' occurrences, there are c*(c-1)//2 pairs.
    ans1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For Case 2 (s > t):
    # We need (P[s-1] - P[t-1]) % M == total_sum % M
    # Let R = total_sum % M.
    # We need P[s-1] % M - P[t-1] % M = R (mod M)
    # Which means P[t-1] % M = (P[s-1] % M - R) % M
    R = total_sum % M
    
    # For each s, we need to count t < s such that rems[t-1] == (rems[s-1] - R) % M
    # However, the condition s > t is symmetric to s < t if we look at the whole set.
    # The number of pairs (s, t) with s > t such that (P[s-1] - P[t-1]) % M == R
    # is the sum over all r: count(r) * count((r - R) % M)
    # BUT, we must exclude the case where s == t (which is not allowed by s != t).
    # If R == 0, then (r - 0) % M == r, so we subtract count(r) and then divide by 2? 
    # No, the logic is simpler:
    # For a fixed remainder r1, the number of pairs (s, t) with s > t is:
    # sum(count(r1) * count((r1 - R) % M))
    # Wait, the "s > t" logic is: for every pair {i, j} with i < j, 
    # one is the "clockwise" distance from i to j, the other is from j to i.
    # Total pairs (s, t) is the sum over all r: count(r) * count((r - R) % M)
    # If R == 0, we must subtract the cases where s == t, which is sum(count(r)).
    
    # Let's refine:
    # We want pairs (s, t) such that dist(s, t) % M == 0.
    # dist(s, t) = (P[t-1] - P[s-1]) % total_sum
    # If s < t: dist = P[t-1] - P[s-1]
    # If s > t: dist = total_sum - (P[s-1] - P[t-1])
    
    # Condition: 
    # s < t: P[t-1] % M == P[s-1] % M
    # s > t: P[s-1] % M - P[t-1] % M == total_sum % M (mod M)
    
    # Let rems be the list of P_i % M.
    # Let C be the Counter of rems.
    # For s < t, we count pairs (i, j) with i < j and rems[i] == rems[j].
    # This is sum(c*(c-1)//2 for c in C.values()).
    # For s > t, we count pairs (i, j) with i < j and (rems[j] - rems[i]) % M == (total_sum % M).
    # Let R = total_sum % M.
    # We need rems[i] == (rems[j] - R) % M.
    # For a fixed j, the number of i < j is the number of times (rems[j] - R) % M appeared before.
    # This is harder without loops. Let's use the property:
    # Total pairs (s, t) with s != t is:
    # sum_{r} (count(r) * count((r - R) % M)) 
    # If R == 0, we must subtract N because s cannot be t.
    # But this counts all pairs (s, t) regardless of whether s < t or s > t.
    # Let's check:
    # If s < t, dist = P[t-1] - P[s-1]. We want P[t-1] - P[s-1] = 0 mod M.
    # If s > t, dist = total_sum - (P[s-1] - P[t-1]). We want P[s-1] - P[t-1] = total_sum mod M.
    # Let R = total_sum % M.
    # s < t: rems[t-1] == rems[s-1]
    # s > t: rems[s-1] - rems[t-1] == R (mod M)
    
    # Let's use the Counter to calculate this:
    # For each r, the number of s < t is C[r]*(C[r]-1)//2.
    # For each r, the number of s > t is C[r] * C[(r - R) % M].
    # Wait, the s > t case: for a fixed pair {i, j} with i < j, 
    # the distance from j to i is (total_sum - (P[j-1] - P[i-1])).
    # This is 0 mod M iff P[j-1] - P[i-1] == total_sum mod M.
    # Let R = total_sum % M.
    # We need rems[j] - rems[i] == R (mod M).
    # For each r, the number of such pairs is C[r] * C[(r - R) % M].
    # BUT, if R == 0, then rems[j] == rems[i], and we are counting the same pairs as s < t.
    # Actually, if R == 0, then dist(s, t) % M == 0 iff dist(t, s) % M == 0.
    # If R != 0, then for any pair {i, j}, at most one of (s=i, t=j) or (s=j, t=i) can be 0 mod M.
    
    # Correct Logic:
    # 1. Count pairs (i, j) with i < j such that rems[i] == rems[j].
    # 2. Count pairs (i, j) with i < j such that (rems[j] - rems[i]) % M == R.
    # These two sets of pairs are disjoint if R != 0.
    # If R == 0, they are the same set.
    
    # To count pairs (i, j) with i < j and rems[j] - rems[i] == R (mod M) without loops:
    # This is equivalent to counting how many i < j satisfy rems[i] == (rems[j] - R) % M.
    # This can be done by iterating through the list and keeping a running count, 
    # but we can't use loops.
    # However, we can use the fact that we only need the total count.
    # For a fixed r, the number of pairs (i, j) with i < j such that rems[j] - rems[i] == R (mod M)
    # is NOT simply C[r] * C[(r-R)%M] because of the i < j constraint.
    # Actually, it is! Because for any two distinct indices i, j, one must be smaller.
    # If rems[j] - rems[i] == R (mod M) and i < j, that's one pair.
    # If we just sum C[r] * C[(r-R)%M] over all r, we are counting all pairs (s, t) 
    # such that (P[s-1] - P[t-1]) % M == R.
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (total_sum - (P[s-1] - P[t-1])) % M == 0
    
    # Let R = total_sum % M.
    # Condition 1: rems[t-1] == rems[s-1] (with s < t)
    # Condition 2: rems[s-1] - rems[t-1] == R (mod M) (with s > t)
    
    # Let's use the property:
    # For any two distinct indices i, j in {0, ..., N-1}, let i < j.
    # The pair (s=i+1, t=j+1) is valid if rems[j] - rems[i] == 0 (mod M).
    # The pair (s=j+1, t=i+1) is valid if rems[j] - rems[i] == R (mod M).
    
    # If R == 0:
    # Both conditions are the same: rems[j] == rems[i].
    # Each such pair {i, j} gives 2 valid (s, t) pairs.
    # Total = 2 * sum(C[r] * (C[r] - 1) // 2) = sum(C[r] * (C[r] - 1))
    
    # If R != 0:
    # Condition 1: rems[j] == rems[i]. Number of pairs = sum(C[r] * (C[r] - 1) // 2)
    # Condition 2: rems[j] - rems[i] == R (mod M).
    # For each pair {i, j} with i < j, it's either Condition 1, Condition 2, or neither.
    # It cannot be both because R != 0.
    # How many pairs {i, j} with i < j satisfy rems[j] - rems[i] == R (mod M)?
    # This is tricky