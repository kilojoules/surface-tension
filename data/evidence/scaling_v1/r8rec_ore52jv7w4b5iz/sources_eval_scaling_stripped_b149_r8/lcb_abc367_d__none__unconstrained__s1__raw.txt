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
    
    # Let P_i be the prefix sum of A.
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) mod M.
    # The distance from s to t (s > t) is (TotalSum - P_{s-1} + P_{t-1}) mod M.
    # We want distance % M == 0.
    
    # Calculate prefix sums modulo M. 
    # P[i] = sum(A[0...i-1]) % M. P[0] = 0.
    # We use accumulate to avoid loops.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of A modulo M
    total_sum = P[N]
    
    # We are looking for pairs (s, t) with 1 <= s, t <= N and s != t.
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # Note: The distance from s to t is A_s + ... + A_{t-1}.
    # In 0-indexing, that is P[t-1] - P[s-1].
    # Wait, the problem says A_i is distance from i to i+1.
    # Dist(1, 2) = A_1. Dist(1, 3) = A_1 + A_2.
    # Let's redefine: P[i] is sum of A_1...A_i.
    # Dist(s, t) for s < t is P[t-1] - P[s-1].
    # Let's use the prefix sums of A: P = [0, A1, A1+A2, ..., A1+...+AN]
    # Dist(s, t) for s < t is P[t-1] - P[s-1].
    # Actually, the distance from s to t (s < t) is sum(A[s-1]...A[t-2]).
    # Let P[i] = sum(A[0...i-1]) % M.
    # Dist(s, t) = (P[t-1] - P[s-1]) % M.
    # For s < t, we need P[t-1] == P[s-1] (mod M).
    # For s > t, we need (total_sum - P[s-1] + P[t-1]) % M == 0 
    # => P[s-1] - P[t-1] == total_sum (mod M).
    
    # Let's use a more direct approach:
    # Let X_i = P[i-1] for i = 1...N.
    # Pair (s, t) with s < t: X_t - X_s == 0 (mod M) is wrong.
    # Let's re-evaluate:
    # Dist(1, 2) = A_1
    # Dist(1, 3) = A_1 + A_2
    # Dist(s, t) for s < t is sum(A[s-1]...A[t-2]) = P[t-1] - P[s-1]
    # Dist(s, t) for s > t is sum(A[s-1]...A[N-1]) + sum(A[0]...A[t-2])
    # = (P[N] - P[s-1]) + P[t-1]
    
    # Let Y_i = P[i-1] for i = 1...N.
    # s < t: Y_t - Y_s = 0 (mod M)  <-- This is for Dist(s, t) = A_s + ... + A_{t-1}
    # Wait, the sample says Dist(1, 2) = 2 (A_1).
    # So Dist(s, t) for s < t is P[t-1] - P[s-1] where P is prefix sum of A.
    # Let's use: P[i] = sum(A[0...i-1]) % M.
    # s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # s > t: (P[N] - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == P[N] (mod M)
    
    # Let's use the values P[0], P[1], ..., P[N-1]
    # These are the prefix sums modulo M.
    # For each pair (i, j) with 0 <= i < j <= N-1:
    # Pair (s=i+1, t=j+1): Dist is (P[j] - P[i]) % M. (Since Dist(1,2)=A_1, Dist(i+1, j+1)=P[j]-P[i])
    # Pair (s=j+1, t=i+1): Dist is (P[N] - P[j] + P[i]) % M.
    
    # Let's simplify:
    # We have N values: V = [P[0], P[1], ..., P[N-1]]
    # We want pairs (i, j) with i < j such that:
    # 1. (P[j] - P[i]) % M == 0  => P[j] == P[i]
    # 2. (P[N] - P[j] + P[i]) % M == 0 => P[j] - P[i] == P[N] (mod M)
    
    # Note: P[j] is the sum of A[0...j-1].
    # P[0] = 0
    # P[1] = A[0]
    # ...
    # P[N-1] = A[0] + ... + A[N-2]
    # P[N] = A[0] + ... + A[N-1]
    
    # The values we care about are P[0], P[1], ..., P[N-1].
    # Let's call this list 'vals'.
    vals = P[:N]
    counts = Counter(vals)
    
    # For a fixed value v, if it appears c times, it contributes c*(c-1)//2 pairs.
    # This covers P[j] == P[i].
    ans = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For P[j] - P[i] == P[N] (mod M) with i < j:
    # This is trickier because of the i < j constraint.
    # However, we can use the property:
    # Total pairs (i, j) such that P[j] - P[i] == P[N] (mod M)
    # minus pairs where i >= j.
    # But we can just iterate through the prefix sums and maintain a counter.
    
    # To avoid loops, we can use a generator expression with a mutable counter.
    # But we can't use a loop. We can use a trick with a list and a counter.
    # Actually, the condition P[j] - P[i] == P[N] (mod M) for i < j
    # is equivalent to counting pairs (P[i], P[j]) such that P[j] = (P[i] + P[N]) % M.
    
    # Let's use a different approach for the second condition:
    # We want to count pairs (i, j) such that 0 <= i < j <= N-1 and P[j] - P[i] \equiv P[N] \pmod M.
    # This is equivalent to: for each j, count i < j such that P[i] \equiv P[j] - P[N] \pmod M.
    
    # Since we can't use loops, we can use a technique to process the sequence.
    # But wait, the constraint is simply to avoid 'for' and 'while'.
    # We can use a list comprehension that updates a dictionary.
    # In Python, `[counts.update({v: counts[v]+1}) for v in vals]` is a way to loop.
    # But the prompt says "without using any for or while loops". 
    # Usually, this implies using map, filter, reduce, or comprehensions.
    # Even comprehensions are loops. But often "no for loops" means "no for-statements".
    # Let's try to be strictly functional.
    
    # For the second condition: P[j] - P[i] == P[N] (mod M) for i < j.
    # If P[N] == 0 (mod M), then P[j] == P[i], which is already counted.
    # But the problem says s != t. If P[N] == 0, then (P[j] - P[i]) == 0 and (P[N] - P[j] + P[i]) == 0.
    # So each pair (i, j) is counted twice.
    # However, the condition s < t and s > t are distinct.
    # If P[N] % M == 0, then P[j] == P[i] implies both clockwise directions are multiples of M.
    
    # Let's use a more robust method.
    # Total pairs (s, t) is the sum over all v of:
    # count(v) * count((v + P[N]) % M)
    # UNLESS v == (v + P[N]) % M, then it's count(v) * (count(v) - 1).
    # This counts all pairs (s, t) such that Dist(s, t) is a multiple of M.
    # Let's check:
    # Dist(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # Dist(s, t) = (P[N] - P[s-1] + P[t-1]) % M if s > t
    # We want Dist(s, t) == 0 (mod M).
    # If s < t: P[t-1] == P[s-1] (mod M)
    # If s > t: P[s-1] - P[t-1] == P[N] (mod M)
    
    # Let's use the property:
    # For any two distinct indices i, j in {0, ..., N-1}:
    # Either (i < j and P[j] == P[i]) or (i > j and P[i] - P[j] == P[N])
    # OR (i < j and P[j] - P[i] == P[N]) or (i > j and P[i] == P[j])
    
    # Let's just count:
    # For each v in 0...M-1:
    # If P[N] % M == 0:
    #   Each pair {i, j} with P[i] == P[j] == v contributes 2 pairs (s,t) and (t,s).
    #   Contribution: count(v) * (count(v) - 1)
    # If P[N] % M != 0:
    #   Pair (s, t) with s < t: P[t-1] == P[s-1]
    #   Pair (s, t) with s > t: P[s-1] - P[t-1] == P[N]
    #   This is:
    #   Sum_{v} (count(v) * (count(v) - 1) // 2)  <-- for s < t
    #   + Sum_{v} (count(v) * count((v - P[N]) % M)) <-- for s > t
    #   Wait, the second term is: for each s, count t < s such that P[t-1] == P[s-1] - P[N].
    #   This is exactly Sum_{v} count(v) * count((v - P[N]) % M).
    #   But we must be careful if P[N] == 0.
    