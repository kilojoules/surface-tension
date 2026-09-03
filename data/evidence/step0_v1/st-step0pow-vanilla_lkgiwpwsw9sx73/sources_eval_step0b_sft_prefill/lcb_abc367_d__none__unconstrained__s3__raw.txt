```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read input using sys.stdin.read().split() to get all tokens
    # The first two are N and M, the rest are A_i
    data = sys.stdin.read().split()
    if not data:
        return
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))
    
    # Let P_i be the prefix sum of A.
    # P_0 = 0
    # P_i = A_1 + ... + A_i
    # The distance clockwise from s to t:
    # If s < t: distance = P_{t-1} - P_{s-1}
    # If s > t: distance = (P_N - P_{s-1}) + P_{t-1}
    # We want distance % M == 0.
    
    # Let X_i = P_i % M.
    # If s < t: (X_{t-1} - X_{s-1}) % M == 0  => X_{t-1} == X_{s-1}
    # If s > t: (X_N - X_{s-1} + X_{t-1}) % M == 0 => X_{s-1} - X_{t-1} == X_N % M
    
    # Note: The problem defines A_i as distance from i to i+1.
    # Let's use 0-indexed logic for the prefix sums:
    # S = [0, A[0], A[0]+A[1], ..., sum(A)]
    # S_i = sum of A_j for j < i.
    # Distance from s to t (1-indexed):
    # If s < t: S_{t-1} - S_{s-1}
    # If s > t: (S_N - S_{s-1}) + S_{t-1}
    
    # Let Y_i = S_i % M for i = 0 ... N-1.
    # We seek pairs (s, t) with 1 <= s, t <= N, s != t.
    # Case 1: s < t. Condition: Y_{t-1} == Y_{s-1}.
    # Case 2: s > t. Condition: Y_{s-1} - Y_{t-1} == S_N % M (mod M).
    
    # Let Y be the list of prefix sums modulo M: [S_0 % M, S_1 % M, ..., S_{N-1} % M]
    # S_0 = 0.
    # S_i = accumulate(A)
    
    # Using a trick: 
    # Total pairs = (pairs s, t where s < t and Y_{t-1} == Y_{s-1}) 
    #              + (pairs s, t where s > t and Y_{s-1} - Y_{t-1} == S_N % M)
    
    # Let Y = [S_0 % M, ..., S_{N-1} % M]
    # Let C be the frequency map of Y.
    # For a fixed value v, if it appears k times, it contributes k*(k-1)//2 pairs to Case 1.
    # For Case 2: we need Y_{s-1} - Y_{t-1} = S_N % M (mod M).
    # Let Target = S_N % M.
    # We need Y_{s-1} = (Y_{t-1} + Target) % M.
    # Since s > t, for each t, we need to count s > t such that Y_{s-1} is the target.
    # This is tricky with loops. Let's use the property:
    # Total = sum_{v} (count(v) * count((v + Target) % M))
    # But we must exclude cases where s = t (which is already excluded by s < t and s > t)
    # and handle the case where Target == 0 (where s < t and s > t both reduce to Y_s == Y_t).
    
    # Correct Logic:
    # Let Y = [S_0 % M, S_1 % M, ..., S_{N-1} % M]
    # We want pairs (i, j) with 0 <= i, j < N, i != j such that:
    # If i < j: (Y_j - Y_i) % M == 0  => Y_i == Y_j
    # If i > j: (Y_N - Y_i + Y_j) % M == 0 => Y_i - Y_j == Y_N % M (mod M)
    
    # Let Target = sum(A) % M.
    # Let counts = Counter(Y)
    # For each value v in counts:
    #   k = counts[v]
    #   # Pairs (i, j) with i < j and Y_i = Y_j = v: k * (k - 1) // 2
    #   # Pairs (i, j) with i > j and Y_i - Y_j = Target (mod M):
    #   # This is harder because of the i > j constraint.
    
    # Let's reconsider:
    # We want (Y_j - Y_i) % M == 0 for i < j   own OR  (Y_N - Y_i + Y_j) % M == 0 for i > j.
    # This is equivalent to:
    # Find pairs (i, j) such that 0 <= i < j < N and:
    # 1. Y_i == Y_j
    # 2. Y_i == (Y_j + Target) % M
    
    # Note: If Target == 0, then Y_i == Y_j and Y_i == (Y_j + 0) % M are the same.
    # If Target != 0, these two conditions are mutually exclusive.
    
    # To count pairs (i, j) with i < j satisfying Y_i == Z_j:
    # We can iterate through the array and keep track of counts of Y_i seen so far.
    
    # Let's use a different approach to avoid loops:
    # For a fixed Target:
    # Count1 = sum(k * (k - 1) // 2 for k in counts.values())
    # Count2 = sum(counts[v] * counts[(v - Target) % M] for v in counts)
    # Wait, Count2 is for all (i, j) such that Y_i - Y_j = Target.
    # This includes i < j and i > j.
    # Let's use the property:
    # Total = sum_{i < j} [Y_i == Y_j] + sum_{i > j} [Y_i - Y_j == Target (mod M)]
    # Total = sum_{i < j} [Y_i == Y_j] + sum_{j < i} [Y_i - Y_j == Target (mod M)]
    
    # Let's process the array Y and maintain the counts of values encountered.
    # For each Y_j:
    #   Answer += current_counts[Y_j] (this is for i < j and Y_i == Y_j)
    #   Answer += current_counts[(Y_j + Target) % M] (this is for i < j and Y_i == Y_j + Target)
    #   # Wait, the second term is Y_i - Y_j = Target, which is the condition for i > j? 
    #   # No, if i < j, then Y_i is the "past" and Y_j is the "current".
    #   # The condition for s > t (which is i > j in 0-indexed) is Y_i - Y_j = Target (mod M).
    #   # So for a fixed i, we need Y_j = (Y_i - Target) % M for j < i.
    
    # Correct logic:
    # For each j from 0 to N-1:
    #   1. Count i < j such that Y_i == Y_j
    #   2. Count i < j such that Y_i == (Y_j - Target) % M
    #   (If Target == 0, these are the same, so only count once).
    
    # To implement this without loops:
    # We can't use a loop, but we can use map/reduce or list comprehensions.
    # Actually, we can just use the global counts.
    # Let C be the Counter of Y.
    # Total = sum(C[v] * (C[v] - 1) // 2 for v in C) + sum(C[v] * C[(v - Target) % M] for v in C if v != (v - Target) % M) // 1
    # Wait, the second term is sum_{i, j} [Y_i - Y_j = Target]. 
    # This equals sum_{v} C[v] * C[(v - Target) % M].
    # This sum includes pairs where i < j, i > j, and i = j.
    # If Target == 0, it's sum(C[v]^2), which is 2 * sum(C[v](C[v]-1)//2) + N.
    # If Target != 0, it's sum(C[v] * C[(v - Target) % M]). Since Target != 0, v != (v - Target) % M, so i != j is guaranteed.
    # For Target != 0, the sum_{i, j} [Y_i - Y_j = Target] is exactly the number of pairs (i, j) with Y_i - Y_j = Target.
    # Half of these will have i < j and half will have i > j? No, that's only if the distribution is symmetric.
    
    # Let's go back to:
    # Ans = sum_{i < j} [Y_i == Y_j] + sum_{i > j} [Y_i - Y_j == Target (mod M)]
    # Ans = sum_{i < j} [Y_i == Y_j] + sum_{j < i} [Y_i - Y_j == Target (mod M)]
    
    # Let's use the identity:
    # sum_{i < j} [Y_i == Y_j] = (sum_{v} C[v]^2 - N) / 2
    # sum_{j < i} [Y_i - Y_j == Target] = ?
    
    # If Target == 0:
    # Ans = 2 * (sum(C[v]^2 - N) / 2) = sum(C[v]^2) - N.
    
    # If Target != 0:
    # Let S1 = sum_{i < j} [Y_i == Y_j]
    # Let S2 = sum_{j < i} [Y_i - Y_j == Target]
    # We know sum_{all i, j} [Y_i - Y_j == Target] = sum_{v} C[v] * C[(v - Target) % M]
    # Also sum_{all i, j} [Y_i - Y_j == Target] = sum_{i < j} [Y_i - Y_j == Target] + sum_{i > j} [Y_i - Y_j == Target]
    # And sum_{i < j} [Y_i - Y_j == Target] = sum_{i < j} [Y_j - Y_i == -Target]
    # Let Target' = -Target % M.
    # S2 = sum_{i > j} [Y_i - Y_j == Target]
    # S1 = sum_{i < j} [Y_i == Y_j]
    
    # Actually, there is a much simpler way.
    # For any two distinct indices i, j, they form two pairs: (s, t) = (i+1, j+1) and (s, t) = (j+1, i+1).
    # One of them is "clockwise" with distance (Y_j - Y_i) % M, the other is (Y_i - Y_j) % M.
    # We want (Y_j - Y_i) % M == 0 OR (Y_i - Y_j) % M == 0.
    # This is equivalent to Y_i == Y_j.
    # If Y_i == Y_j, then both (i+1, j+1) and (j+1, i+1) are multiples of M.
    # If Y_i != Y_j, then (Y_j - Y_i) % M is some value and (Y_i - Y_j) % M is M - that value.
    # One of these is 0