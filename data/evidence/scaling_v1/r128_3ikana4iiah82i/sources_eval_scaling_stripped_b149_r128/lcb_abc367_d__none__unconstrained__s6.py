```python
import sys
from collections import Counter

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of A modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A[0] % M, P[2] = (A[0] + A[1]) % M, ...
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1]
    
    # Using a list comprehension to build the prefix sums modulo M
    # Since we cannot use loops, we use a trick with a helper function or 
    # we can use the fact that we need the counts of each remainder.
    # However, calculating prefix sums without loops usually requires 
    # itertools.accumulate.
    from itertools import accumulate
    
    # P will contain the cumulative sums of A modulo M
    # P = [0, A[0]%M, (A[0]+A[1])%M, ...]
    # We prepend 0 to represent the starting point (rest area 1)
    P = list(accumulate(A, lambda x, y: (x + y) % M))
    # To include the 0-th prefix sum (distance from 1 to 1), we insert 0 at the start
    # But the problem says s != t, so we are looking for pairs (s, t)
    # If we consider the sequence S = [0] + P, then for 1 <= s < t <= N:
    # distance is (S[t-1] - S[s-1]) % M. 
    # Wait, the indices are:
    # s=1, t=2 -> A[0]
    # s=1, t=3 -> A[0] + A[1]
    # s=N, t=1 -> A[N-1]
    # This is a circular array. The distance from s to t is:
    # If s < t: sum(A[s-1] ... A[t-2])
    # If s > t: sum(A[s-1] ... A[N-1]) + sum(A[0] ... A[t-2])
    
    # Let TotalSum = sum(A) % M
    # Let P[i] be the prefix sum modulo M: P[i] = sum(A[0]...A[i-1]) % M
    # For s < t: distance is (P[t-1] - P[s-1]) % M
    # For s > t: distance is (TotalSum - P[s-1] + P[t-1]) % M
    
    # Let's simplify:
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # AND (TotalSum - P[s-1] + P[t-1]) % M == 0 for s > t
    
    # Let's define S = [0] + P (where P is the accumulate result)
    # S has N+1 elements. S[0]=0, S[1]=A[0]%M, ..., S[N]=sum(A)%M
    # For 1 <= s < t <= N:
    # Distance is (S[t-1] - S[s-1]) % M. 
    # This is 0 if S[t-1] == S[s-1].
    # For 1 <= t < s <= N:
    # Distance is (S[N] - S[s-1] + S[t-1]) % M.
    # This is 0 if S[s-1] - S[t-1] == S[N] % M.
    
    # Let's use a more robust approach:
    # The distance from s to t is (P[t-1] - P[s-1]) % M if s < t
    # and (P[N] - P[s-1] + P[t-1]) % M if s > t.
    # Note: P[i] is the sum of first i elements of A.
    
    # Let's redefine: 
    # Let Pref = [0] + list(accumulate(A, lambda x, y: (x + y) % M))
    # Pref[i] is the distance from area 1 to area i+1.
    # For any pair (s, t) with s != t:
    # If s < t: dist = (Pref[t-1] - Pref[s-1]) % M
    # If s > t: dist = (Pref[N] - Pref[s-1] + Pref[t-1]) % M
    
    # We want dist % M == 0.
    # Case 1: s < t and Pref[t-1] == Pref[s-1]
    # Case 2: s > t and Pref[s-1] - Pref[t-1] == Pref[N] % M
    
    # Let's count occurrences of each value in Pref[0...N-1]
    # Note: Pref[N] is the total sum.
    # The values we care about are Pref[0], Pref[1], ..., Pref[N-1].
    # Let these be V = [Pref[0], ..., Pref[N-1]].
    # We want pairs (i, j) such that:
    # 1. i < j and V[i] == V[j]
    # 2. i > j and (V[i] - V[j]) % M == Pref[N] % M
    
    # Let Total = Pref[N]
    # For a fixed value v, let count(v) be the number of times v appears in V.
    # The number of pairs (i, j) with i < j and V[i] == V[j] is:
    # sum( count(v) * (count(v) - 1) / 2 )
    
    # For the second case: i > j and V[i] - V[j] == Total (mod M)
    # This is equivalent to V[i] - Total == V[j] (mod M)
    # For each v, we want to find how many j < i have V[j] == (v - Total) % M.
    # This is tricky because of the i > j constraint.
    # Actually, we can just iterate through all possible values of v in 0...M-1.
    # For a specific v, let C(v) be the count of v in V.
    # The number of pairs (i, j) with i != j such that (V[i] - V[j]) % M == Total % M is:
    # If Total % M == 0: it's the same as V[i] == V[j], which is C(v)*(C(v)-1)
    # If Total % M != 0: it's C(v) * C((v - Total) % M)
    
    # Wait, the condition is:
    # If s < t: (Pref[t-1] - Pref[s-1]) % M == 0  => Pref[t-1] == Pref[s-1]
    # If s > t: (Pref[N] - Pref[s-1] + Pref[t-1]) % M == 0 => Pref[s-1] - Pref[t-1] == Pref[N] % M
    
    # Let V = [Pref[0], ..., Pref[N-1]]
    # We want pairs (i, j) with 0 <= i, j < N and i != j such that:
    # (i < j and V[i] == V[j]) OR (i > j and V[i] - V[j] == Pref[N] % M)
    
    # Let's analyze the second condition: i > j and V[i] - V[j] == Total % M
    # This is equivalent to: i > j and V[j] == (V[i] - Total) % M
    
    # If Total % M == 0:
    # Condition 1: i < j and V[i] == V[j]
    # Condition 2: i > j and V[i] == V[j]
    # Total pairs: sum( C(v) * (C(v) - 1) )
    
    # If Total % M != 0:
    # Condition 1: i < j and V[i] == V[j]
    # Condition 2: i > j and V[i] - V[j] == Total % M
    # These two conditions are mutually exclusive because V[i] == V[j] and V[i] - V[j] == Total % M
    # cannot both be true if Total % M != 0.
    # So we just need to sum the counts.
    # For Condition 1: sum( C(v) * (C(v) - 1) / 2 )
    # For Condition 2: we need to count pairs (i, j) with i > j and V[j] == (V[i] - Total) % M.
    # This is sum_{i=1 to N-1} (count of (V[i] - Total) % M in V[0...i-1])
    
    # However, we can't use loops. We can use a different approach for Condition 2.
    # The total number of pairs (i, j) with i != j such that (V[i] - V[j]) % M == Total % M is:
    # sum( C(v) * C((v - Total) % M) )
    # Since Total % M != 0, V[i] cannot be equal to (V[i] - Total) % M.
    # Thus, for any pair (i, j) such that (V[i] - V[j]) % M == Total % M,
    # either (i < j and (V[j] - V[i]) % M == -Total % M) 
    # or (i > j and (V[i] - V[j]) % M == Total % M).
    
    # Let's use the property:
    # Total pairs (i, j) with i != j such that (V[i] - V[j]) % M == Total % M is:
    # sum_{v=0 to M-1} C(v) * C((v - Total) % M)
    # Each such pair (i, j) satisfies EITHER (i > j and V[i]-V[j] == Total) 
    # OR (i < j and V[j]-V[i] == -Total).
    # But we want (i > j and V[i]-V[j] == Total).
    # This is exactly the same as saying (i, j) is a pair where the first is the 't' and second is 's'.
    # Actually, the simplest way:
    # The total number of pairs (s, t) is the number of pairs (i, j) with 0 <= i, j < N, i != j
    # such that the clockwise distance is 0 mod M.
    # Clockwise distance from s to t is:
    # (Pref[t-1] - Pref[s-1]) % M if s < t
    # (Pref[N] - Pref[s-1] + Pref[t-1]) % M if s > t
    
    # Let's use the property:
    # Dist(s, t) = (Pref[t-1] - Pref[s-1]) % M
    # But if s > t, we must add Pref[N] to the difference.
    # Dist(s, t) = (Pref[t-1] - Pref[s-1] + (Pref[N] if s > t else 0)) % M
    
    # This is equivalent to:
    # Find pairs (i, j) with 0 <= i, j < N, i != j such that:
    # 1. i < j and Pref[j] == Pref[i]
    # 2. i > j and Pref[j] - Pref[i] == -Pref[N] % M  => Pref[i] - Pref[j] == Pref[N] % M
    
    # Let Total = Pref[N] % M.
    # We want:
    # Case 1: i < j and Pref[i] == Pref[j]
    # Case 2: i > j and Pref[i] - Pref[j] == Total
    
    