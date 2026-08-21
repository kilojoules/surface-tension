The constraint to avoid explicit `for` and `while` loops forces a functional programming style, which encourages the use of vectorized-like operations and higher-order functions. In Python, this often leads to more concise code by utilizing list comprehensions and `itertools` to handle data transformations and aggregations. For this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances and `collections.Counter` to efficiently count the occurrences of each remainder modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use a trick to unpack the iterator
    # We can't use a for loop, but we can use a list comprehension 
    # or map to consume the input.
    
    # To handle the input without a for loop, we convert the map to a list
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Prefix sums: P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A[0], A[0]+A[1], ...]
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # accumulate(A) gives [A0, A0+A1, ...]
    # We prepend 0 to represent the starting point (rest area 1)
    P = list(accumulate(A))
    total_sum = P[-1]
    
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # and (total_sum - P[s-1] + P[t-1]) % M == 0 for s > t
    
    # Let R[i] = P[i-1] % M (with P[0] = 0)
    # For s < t: R[t-1] == R[s-1]
    # For s > t: R[t-1] == (R[s-1] - total_sum) % M
    
    # Create the list of remainders R
    # R[0] = 0, R[1] = A[0]%M, R[2] = (A[0]+A[1])%M ...
    # Note: P already contains sums from 1 to N. 
    # We need the remainders of [0, P[0], P[1], ..., P[N-2]]
    # because the distance from s to t (s < t) is P[t-1] - P[s-1].
    # The possible values for s are 1 to N.
    # The possible values for t are 1 to N.
    
    # R contains P[i] % M for i in 0...N-1
    # P[0] is A[0], P[N-1] is total_sum.
    # We need the sequence: 0, A[0], A[0]+A[1]... (up to N terms)
    R = [0] + [p % M for p in P[:-1]]
    
    # Count frequencies of each remainder
    counts = Counter(R)
    
    # For a fixed s, we need t such that:
    # If s < t: R[t-1] % M == R[s-1] % M
    # If s > t: R[t-1] % M == (R[s-1] - total_sum) % M
    
    # Let target1 = R[s-1]
    # Let target2 = (R[s-1] - total_sum) % M
    
    # For each s, the number of t's is:
    # (count of target1) - 1  <-- the -1 is because s != t
    # BUT, this is tricky because of the s < t and s > t conditions.
    
    # Let's re-evaluate:
    # A pair (s, t) is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (total_sum - P[s-1] + P[t-1]) % M == 0
    
    # Let R[i] = P[i] % M for i = 0...N-1 (where P[0]=0)
    # Condition 1: R[t-1] == R[s-1] where s < t
    # Condition 2: R[t-1] == (R[s-1] - total_sum) % M where s > t
    
    # Let, for each remainder r, C[r] be the number of times it appears in R.
    # The number of pairs (s, t) with s < t and R[s-1] == R[t-1] is:
    # sum( C[r] * (C[r] - 1) // 2 )
    
    # The number of pairs (s, t) with s > t and R[t-1] == (R[s-1] - total_sum) % M:
    # This is harder because of the s > t constraint.
    # However, notice that if total_sum % M == 0, then Condition 2 is R[t-1] == R[s-1].
    # If total_sum % M != 0, then R[t-1] and R[s-1] must be different.
    
    # Let's use the property:
    # Total pairs = sum_{s=1 to N} (count of t such that dist(s,t) % M == 0)
    # For a fixed s, t is valid if:
    # t > s and R[t-1] == R[s-1]
    # t < s and R[t-1] == (R[s-1] - total_sum) % M
    
    # Let', for each s, r = R[s-1].
    # We need t > s with R[t-1] = r AND t < s with R[t-1] = (r - total_sum) % M.
    
    # Let's use the counts:
    # For a specific remainder r, let indices be i_1, i_2, ..., i_k.
    # For each i_j, the number of t > s is (k - j).
    # For the second condition, let r' = (r - total_sum) % M.
    # For each i_j, we need t < s such that R[t-1] = r'.
    # This is the number of indices of r' that are less than i_j.
    
    # To avoid loops, we can use a list comprehension to sum these up.
    # But we can simplify:
    # Total = sum_{r} [ (C[r]*(C[r]-1)//2) + sum_{s: R[s-1]=r} (count of t < s with R[t-1]=r') ]
    
    # Let's use the fact that we can iterate through the list R once.
    # We need to track how many times each remainder has appeared so far.
    # Since we can't use for loops, we can't maintain a running state easily.
    # However, we can use a list comprehension to calculate the "t < s" part.
    
    # Let', for each r, the list of indices where R[i] == r be Ind[r].
    # The number of t < s such that R[t-1] = r' is the count of indices in Ind[r'] 
    # that are smaller than the current index in Ind[r].
    
    # Actually, a simpler way:
    # For every pair of remainders (r, r') such that r' = (r - total_sum) % M:
    # We want to count pairs (s, t) such that:
    # (s < t and R[s-1] == r and R[t-1] == r) OR
    # (s > t and R[s-1] == r and R[t-1] == r')
    
    # If r == r' (which happens if total_sum % M == 0):
    # We need s != t and R[s-1] == R[t-1]. 
    # For each r, this is C[r] * (C[r] - 1).
    
    # If r != r':
    # We need (s < t and R[s-1] == r and R[t-1] == r) OR (s > t and R[s-1] == r and R[t-1] == r')
    # This is (C[r]*(C[r]-1)//2) + (C[r'] * C[r] - (count of s < t where R[s-1]=r and R[t-1]=r'))
    # This is getting complex. Let's use the most direct logic:
    # For each s, we need t != s such that dist(s, t) % M == 0.
    # dist(s, t) = (P[t-1] - P[s-1]) if s < t else (total_sum - P[s-1] + P[t-1])
    
    # Let r_s = P[s-1] % M.
    # t is valid if:
    # 1. t > s and P[t-1] % M == r_s
    # 2. t < s and P[t-1] % M == (r_s - total_sum) % M
    
    # Let's use the property:
    # Total = sum_{r} [ C[r]*(C[r]-1)//2 ]  <-- this covers s < t and R[s-1]==R[t-1]
    #       + sum_{s=1 to N} [ count of t < s such that R[t-1] == (R[s-1] - total_sum) % M ]
    
    # To calculate the second term without a loop:
    # We can use a list comprehension with a dictionary/counter that we update? 
    # No, that's a side effect. 
    # We can use a list comprehension to find all indices for each remainder.
    # But we can just use the counts:
    # For a fixed r, let r' = (r - total_sum) % M.
    # We want to count pairs (t, s) such that t < s, R[t-1] = r', R[s-1] = r.
    # If r == r', this is C[r]*(C[r]-1)//2.
    # If r != r', we can't simply use C[r]*C[r']. We need t < s.
    
    # Wait, if we sum over all s:
    # Ans = sum_{s=1 to N} ( (count of t > s with R[t-1] == R[s-1]) 
    #                      + (count of t < s with R[t-1] == (R[s-1] - total_sum) % M) )
    
    # Let r = R[s-1] and r' = (r - total_sum) % M.
    # The first term is (C[r] - 1 - (count of t < s with R[t-1] == r)).
    # The second term is (count of t < s with R[t-1] == r').
    
    # Total = sum_{s=1 to N} [ C[R[s-1]] - 1 - (count t < s, R[t-1]==R[s-1]) + (count t < s, R[t-1]==r') ]
    # Total = sum_{r} (C[r] * (C[r] - 1)) - sum_{r} (C[r]*(C[r]+1)//2) ... no.
    
    # Let's use: sum_{s=1 to N} (C[R[s-1]] - 1) 
    # this is sum_{r} C[r]*(C[r]-1).
    # Then we adjust for the t < s part.
    # For a fixed r, the sum of (count t < s with R[t-1] == r) over all s with R[s-1] == r
    # is 0 + 1 + 2 + ... + (C[r]-1) = C[r]*(C