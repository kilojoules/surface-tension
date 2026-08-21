The constraint to avoid explicit loops forces a shift from imperative style to a functional paradigm, emphasizing data transformations over state mutation. In idiomatic Python, this encourages the use of high-level abstractions like `map`, `functools.reduce`, and list comprehensions, which can often be more concise and expressive for sequence processing. For this problem, I will use `itertools.accumulate` to compute prefix sums of the distances and `collections.Counter` to efficiently count occurrences of remainders modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Compute prefix sums modulo M
    # P = [0, A1, A1+A2, ..., A1+...+AN]
    # We only need the first N prefix sums for the remainders.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of all A_i modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # Case 2: s > t
    # (total_sum_mod - P[s-1] + P[t-1]) % M == 0 => P[s-1] % M == (P[t-1] + total_sum_mod) % M
    
    # Count occurrences of each remainder in P[0...N-1]
    # P[N] is the total sum, we only consider indices 0 to N-1 for s and t.
    counts = Counter(P[:N])
    
    # For Case 1: For each remainder r, there are counts[r] * (counts[r] - 1) / 2 pairs.
    # However, the problem asks for pairs (s, t), and s != t.
    # If s < t, we need P[t-1] == P[s-1].
    # If s > t, we need P[s-1] == (P[t-1] + total_sum_mod) % M.
    
    # Let's calculate Case 1 (s < t)
    # For each remainder r, we have count C. The number of pairs (s, t) with s < t is C*(C-1)//2.
    # But we can just iterate over the counts and sum them up.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For Case 2 (s > t):
    # We need P[s-1] % M == (P[t-1] + total_sum_mod) % M.
    # Let r_t = P[t-1] % M and r_s = P[s-1] % M.
    # We need r_s == (r_t + total_sum_mod) % M.
    # For a fixed r_t, there are counts[r_t] choices for t and counts[(r_t + total_sum_mod) % M] choices for s.
    # This includes cases where s = t if total_sum_mod % M == 0, but the constraint is s != t.
    # Since we specifically need s > t, we can't simply multiply.
    
    # Let's refine:
    # Total pairs (s, t) with s != t is the goal.
    # Distance(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # Distance(s, t) = (P[N] - P[s-1] + P[t-1]) % M if s > t
    
    # Let X = P[s-1] and Y = P[t-1].
    # If s < t: (Y - X) % M == 0  => Y % M == X % M
    # If s > t: (total_sum_mod - X + Y) % M == 0 => X % M == (Y + total_sum_mod) % M
    
    # Let's use a different approach to avoid loops and recursion:
    # For every pair of indices i, j in {0, ..., N-1} with i != j:
    # If i < j, we check if P[j] % M == P[i] % M
    # If i > j, we check if P[i] % M == (P[j] + total_sum_mod) % M
    
    # This can be solved by:
    # 1. Sum_{r} (count[r] * (count[r]-1) // 2)  <-- This covers s < t
    # 2. Sum_{r} (count[r] * count[(r + total_sum_mod) % M]) 
    #    BUT we must subtract the cases where s > t is actually s < t or s = t.
    #    Actually, the second sum is simply: for each t, how many s > t satisfy the condition.
    
    # Let's use the property:
    # Total = Sum_{t=1 to N} (count of s < t such that dist(s,t)%M==0) 
    #       + Sum_{t=1 to N} (count of s > t such that dist(s,t)%M==0)
    
    # For a fixed t (index j = t-1):
    # s < t (index i < j): P[j] % M == P[i] % M
    # s > t (index i > j): P[i] % M == (P[j] + total_sum_mod) % M
    
    # Let's use a frequency map for all P[0...N-1].
    # For each j from 0 to N-1:
    #   Ways to pick i < j: (number of times P[j] appeared before index j)
    #   Ways to pick i > j: (number of times (P[j] + total_sum_mod)%M appears after index j)
    
    # To do this without loops, we can use the total counts and subtract the "before" counts.
    # Let target(j) = (P[j] + total_sum_mod) % M.
    # Total ways = Sum_{j=0 to N-1} [ (count of P[j] in P[0...j-1]) + (count of target(j) in P[j+1...N-1]) ]
    
    # Count of P[j] in P[0...j-1] summed over all j is Sum (C*(C-1)//2)
    # Count of target(j) in P[j+1...N-1] is:
    # Sum_{j=0 to N-1} [ TotalCount(target(j)) - (count of target(j) in P[0...j]) ]
    
    # Let's implement this using map/sum.
    
    # 1. s < t:
    ans1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # 2. s > t:
    # We need to calculate Sum_{j=0 to N-1} (Count(target(j)) - Count of target(j) in P[0...j])
    # Let target_j = (P[j] + total_sum_mod) % M
    # Total = Sum_{j=0 to N-1} Count(target_j) - Sum_{j=0 to N-1} (Count of target_j in P[0...j])
    
    # The first part:
    # Sum_{j=0 to N-1} counts[(P[j] + total_sum_mod) % M]
    # The second part:
    # We need the sum of counts of target_j in the prefix.
    # We can use a running count. Since we can't use loops, we use a trick with 
    # a custom function in reduce or a list comprehension with a mutable object.
    # However, the constraint says "no explicit loops", and "reduce" is allowed.
    
    # Let's use a more direct approach for s > t:
    # For each remainder r, let C(r) be the count of r in P[0...N-1].
    # The number of pairs (s, t) with s > t such that P[s-1] % M == (P[t-1] + total_sum_mod) % M is:
    # Sum_{j=0 to N-1} [ (Count of (P[j] + total_sum_mod)%M in P[0...N-1]) - (Count of (P[j] + total_sum_mod)%M in P[0...j]) ]
    
    # Let's simplify the s > t condition:
    # We want pairs (i, j) such that i > j and P[i] % M == (P[j] + total_sum_mod) % M.
    # This is equivalent to: for each j, count i > j such that P[i] % M == target_j.
    
    # To compute this without loops, we can use the fact that:
    # Sum_{j=0 to N-1} (Count of target_j in P[j+1...N-1])
    # = Sum_{r=0 to M-1} [ Sum_{j: P[j]%M == r} (Count of (r + total_sum_mod)%M in P[j+1...N-1]) ]
    
    # If total_sum_mod % M == 0:
    # Then target_j = P[j]. We are looking for i > j such that P[i] == P[j].
    # This is exactly the same as s < t, which is C*(C-1)//2.
    
    # If total_sum_mod % M != 0:
    # Let r1 = P[j] % M and r2 = (P[j] + total_sum_mod) % M.
    # Note that r1 != r2.
    # We want to count pairs (j, i) with j < i and P[i] % M == r2 and P[j] % M == r1.
    # This is NOT simply C(r1)*C(r2) because of the j < i constraint.
    # Wait, the problem says "minimum number of steps to walk clockwise from s to t".
    # If s < t, distance is P[t-1] - P[s-1].
    # If s > t, distance is (P[N] - P[s-1]) + P[t-1].
    # These are the ONLY two cases since s != t.
    
    # Let's re-evaluate s > t:
    # Distance = (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    # P[s-1] % M == (P[t-1] + total_sum_mod) % M
    
    # Let r_t = P[t-1] % M and r_s = P[s-1] % M.
    # We need r_s == (r_t + total_sum_mod) % M.
    # For a fixed pair of remainders (r_t, r_s) that satisfy the condition:
    # We need to count pairs (s, t) such that s > t, P[s-1] % M = r_s, and P[t-1] % M = r_t.
    
    # Let's use the property:
    # Total pairs (s, t) with s != t such that dist(s, t) % M == 0 is:
    # Sum_{j=0 to N-1} [ (count of i < j such that P[i] == P[j]) + (count of i > j such that P[i] == (P[j] + total_sum_mod