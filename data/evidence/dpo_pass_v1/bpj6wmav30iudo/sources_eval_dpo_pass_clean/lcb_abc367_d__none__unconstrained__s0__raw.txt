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
    # We'll convert the map object to a list to allow indexing
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Let S[i] be the distance from rest area 1 to rest area i+1
    # S[0] = 0
    # S[1] = A[0]
    # S[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is S[t-1] - S[s-1]
    # The distance from s to t (s > t) is (Total_Sum - S[s-1]) + S[t-1]
    
    # Calculate prefix sums modulo M
    # S will have N elements: S[0]...S[N-1]
    # S[i] is the distance from area 1 to area i+1
    S = list(accumulate(A, lambda x, y: (x + y) % M))
    
    # To handle the "circle", we need the distance from 1 to 1 (0) 
    # and the total sum of A modulo M.
    # Let's create a list of remainders for each starting point relative to area 1.
    # Position 1: 0
    # Position 2: A[0] % M
    # Position 3: (A[0] + A[1]) % M ...
    # Position N: (A[0] + ... + A[N-2]) % M
    
    # The distance from s to t is (S[t-1] - S[s-1]) % M
    # We want (S[t-1] - S[s-1]) % M == 0, which means S[t-1] % M == S[s-1] % M
    
    # Let's redefine S to include the starting point 0
    # S_all = [0, A[0], A[0]+A[1], ..., A[0]...A[N-2]]
    # Note: A[N-1] is the distance from N back to 1.
    
    # We only care about the first N-1 prefix sums for the "internal" distances
    # But the problem asks for any s, t.
    # Let P[i] be the distance from area 1 to area i.
    # P[1] = 0
    # P[2] = A[1]
    # ...
    # P[N] = A[1] + ... + A[N-1]
    # Distance s -> t (clockwise):
    # If s < t: P[t] - P[s]
    # If s > t: (P[N] + A[N]) - P[s] + P[t]
    
    # Let Total = sum(A) % M
    # We want (P[t] - P[s]) % M == 0 if s < t
    # We want (Total - P[s] + P[t]) % M == 0 if s > t
    
    # Let's simplify:
    # Let x_i = P[i] % M.
    # We seek pairs (s, t) such that:
    # 1. s < t and x_t == x_s
    # 2. s > t and x_t - x_s == -Total (mod M) => x_s - x_t == Total (mod, M)
    
    # Let's calculate P for all i from 1 to N
    # P = [0] + list(accumulate(A[:-1]))
    # But wait, A is given as A_1...A_N where A_i is i -> i+1.
    # So P[1] = 0, P[2] = A[0], P[3] = A[0]+A[1]...
    
    # Correct P sequence:
    P = [0] + list(accumulate(A[:N-1], lambda x, y: (x + y) % M))
    Total = sum(A) % M
    
    # Count occurrences of each remainder
    counts = Counter(P)
    
    # For a fixed remainder r, if there are 'c' positions with P[i] == r:
    # The number of pairs (s, t) with s < t and P[s] == P[t] is c * (c - 1) // 2.
    # However, we need to consider s > t as well.
    
    # Let's use the property:
    # Pair (s, t) is valid if:
    # (P[t] - P[s]) % M == 0  where s < t
    # (P[t] - P[s] + Total) % M == 0 where s > t
    
    # This is equivalent to:
    # s < t: P[s] == P[t]
    # s > t: P[s] == (P[t] + Total) % M
    
    # Let',s be the set of indices.
    # Total pairs = Sum_{r=0 to M-1} (count[r] * count[r]) 
    # But we must exclude s == t.
    # And we must handle the Total offset.
    
    # Let's evaluate:
    # For each s, we need t such that:
    # If t > s, P[t] = P[s]
    # If t < s, P[t] = (P[s] - Total) % M
    
    # Let's use the counts:
    # For a fixed s, the number of t's is:
    # (count[P[s]] - 1)  <-- this counts all t where P[t] == P[s], including t < s and t > s
    # But we only want t > s for P[t] == P[s].
    # And we want t < s for P[t] == (P[s] - Total) % M.
    
    # Let's reconsider:
    # A pair (s, t) is valid if:
    # 1. s < t AND P[s] == P[t]
    # 2. s > t AND P[s] == (P[t] + Total) % M
    
    # Let's sum over all r:
    # For a specific r, let indices be i_1, i_2, ..., i_c.
    # Pairs (i_j, i_k) with j < k satisfy condition 1. There are c*(c-1)//2 such pairs.
    # For condition 2: s > t and P[s] == (P[t] + Total) % M.
    # This means P[t] == (P[s] - Total) % M.
    # For a fixed s, the number of t < s such that P[t] == (P[s] - Total) % M.
    
    # This looks like we can iterate through P once. 
    # Since we can't use loops, we use a list comprehension and a helper.
    # But we can't maintain state in a comprehension easily.
    
    # Alternative:
    # Total count = Sum_{r} (count[r] * count[(r - Total) % M])
    # This counts pairs (s, t) where P[s] == (P[t] + Total) % M.
    # If Total == 0:
    # The condition is P[s] == P[t]. 
    # For any s != t, if P[s] == P[t], then (s < t and P[s]==P[t]) OR (s > t and P[s]==P[t]).
    # So if Total == 0, answer is Sum(c * (c-1)) for all c in counts.
    # If Total != 0:
    # Condition 1: s < t and P[s] == P[t]
    # Condition 2: s > t and P[s] == P[t] + Total
    # Notice that if Total != 0, then P[s] == P[t] and P[s] == P[t] + Total cannot both be true.
    # The number of pairs (s, t) with s != t such that P[s] == P[t] is Sum(c * (c-1)).
    # Half of these have s < t.
    # The number of pairs (s, t) such that P[s] == (P[t] + Total) % M is Sum(count[r] * count[(r - Total) % M]).
    
    # Let',s be the set of indices.
    # We want count {(s, t) | s < t, P[s] == P[t]} + {(s, t) | s > t, P[s] == P[t] + Total}
    # Let's use the property:
    # Sum_{s < t, P[s] == P[t]} 1  =  (Sum_{r} c_r * (c_r - 1) // 2)
    # Sum_{s > t, P[s] == P[t] + Total} 1 = ?
    
    # Let's use a different approach.
    # For every pair (s, t) with s != t:
    # Let d(s, t) be the clockwise distance.
    # d(s, t) = (P[t] - P[s]) % Total_Sum_of_A
    # We want d(s, t) % M == 0.
    # d(s, t) = (P[t] - P[s]) % (Sum A)
    # This is a bit complex. Let's use the simpler logic:
    # Clockwise distance from s to t:
    # If s < t: Dist = P[t] - P[s]
    # If s > t: Dist = (P[N] + A[N-1]) - P[s] + P[t]
    # where P[i] is the distance from 1 to i.
    # P[1]=0, P[2]=A[0], ..., P[N]=A[0]+...+A[N-2].
    # Total_Sum = A[0] + ... + A[N-1].
    
    # We want Dist % M == 0.
    # s < t: (P[t] - P[s]) % M == 0  => P[t] % M == P[s] % M
    # s > t: (Total_Sum - P[s] + P[t]) % M == 0 => P[s] - P[t] == Total_Sum % M
    
    # Let x_i = P[i] % M.
    # Ans = Sum_{i < j, x_i == x_j} 1  +  Sum_{i > j, x_i - x_j == Total % M} 1
    
    # Let's calculate the first part:
    # For each r, c_r is the number of times r appears in P.
    # Part 1 = Sum(c_r * (c_r - 1) // 2)
    
    # For the second part:
    # We need i > j such that x_i - x_j == Total % M.
    # This is a convolution-like sum.
    # However, we can't use loops.
    # We can use a list comprehension to sum:
    # Part 2 = Sum(count[r] * count[(r - Total) % M]) 
    # But this counts all (i, j) such that x_i - x_j == Total % M, regardless of i > j.
    
    # Let',s and x_s be the sequence.
    # Let's use the fact that we can't use loops but we can use map/sum.
    # To handle i > j, we can't easily without a loop or a very clever trick.
    # Wait, if we sum over all r: count[r] * count[(r - Total) % M], 