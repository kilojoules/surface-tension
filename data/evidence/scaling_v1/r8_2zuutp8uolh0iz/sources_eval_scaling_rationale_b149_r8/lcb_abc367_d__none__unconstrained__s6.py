The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem without `for` or `while` loops, I will use `itertools.accumulate` to calculate prefix sums of the distances, and `collections.Counter` combined with list comprehensions to count the occurrences of each remainder modulo $M$.

The distance from rest area $s$ to $t$ clockwise is $(P_t - P_s) \pmod L$ if $s < t$, and $(L - (P_s - P_t)) \pmod L$ if $s > t$, where $P_i$ is the prefix sum of distances and $L$ is the total perimeter. Both cases simplify to checking if $P_t \equiv P_s \pmod M$ when considering the distance modulo $M$, provided we handle the wrap-around correctly. Specifically, the distance from $s$ to $t$ is a multiple of $M$ if and only if $P_t \equiv P_s \pmod M$ (for $s < t$) or $(P_t + L) \equiv P_s \pmod M$ (for $s > t$).

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
    
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # The distance from s to t (s < t) is P[t-1] - P[s-1]
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # P_vals will have N elements: distance from area 1 to area 1, 2, ..., N
    P_vals = list(accumulate(A, lambda x, y: (x + y) % M))
    # We need the distance from area 1 to area 1, which is 0
    # So we prepend 0 and take the first N elements
    P = [0] + P_vals
    P = P[:N]
    
    # Total distance L modulo M
    L_mod_M = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(P)
    
    # For a fixed s, we want to find t != s such that:
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # If s > t: (L_mod_M + P[t-1] - P[s-1]) % M == 0 => P[t-1] % M == (P[s-1] - L_mod_M) % M
    
    # Let C[r] be the number of i in {0, ..., N-1} such that P[i] % M == r
    # For each s (represented by its remainder r = P[s-1] % M):
    # The number of t > s such that P[t-1] % M == r is (count of r) - (index of s in sorted P)
    # This is tricky with counts. Let's use the property:
    # Total pairs (s, t) with s < t and P[t-1] == P[s-1] (mod M) is sum(C[r] * (C[r]-1) // 2)
    # Total pairs (s, t) with s > t and P[t-1] == (P[s-1] - L_mod_M) (mod M)
    # is sum(C[r] * C[(r - L_mod_M) % M]) 
    # BUT we must exclude cases where s == t, which is already handled by s > t.
    # However, if L_mod_M == 0, then (r - L_mod_M) % M == r, and we are counting 
    # pairs (s, t) where s > t and P[s-1] == P[t-1].
    
    # Let's refine:
    # 1. Pairs (s, t) with s < t:
    #    For each remainder r, there are C[r] indices. 
    #    Number of pairs is C[r] * (C[r] - 1) // 2.
    # 2. Pairs (s, t) with s > t:
    #    We need P[t-1] % M == (P[s-1] - L_mod_M) % M.
    #    Let r_s = P[s-1] % M and r_t = P[t-1] % M.
    #    We need r_t == (r_s - L_mod_M) % M.
    #    For a fixed r_s, there are C[r_s] choices for s and C[(r_s - L_mod_M) % M] choices for t.
    #    This counts all pairs (s, t) such that the condition holds.
    #    We must subtract cases where s <= t.
    #    Wait, the condition s > t is strict.
    #    Let's use: Total = sum_{r} (C[r] * C[(r - L_mod_M) % M])
    #    This sum includes pairs where s > t, s < t, and s = t.
    #    Specifically, it counts pairs (s, t) such that dist(s, t) is a multiple of M.
    #    If L_mod_M == 0, then dist(s, t) is a multiple of M iff P[s-1] == P[t-1] (mod M).
    #    In that case, for each r, we have C[r] choices for s and C[r] for t.
    #    Total pairs is sum(C[r]^2), then subtract N (where s=t).
    #    If L_mod_M != 0, then s=t implies dist(s, t) = 0, which is a multiple of M.
    #    But the problem says s != t.
    #    If L_mod_M != 0, then P[t-1] == (P[s-1] - L_mod_M) % M and P[t-1] == P[s-1] % M 
    #    cannot both be true.
    
    # Correct Logic:
    # A pair (s, t) with s != t is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (L_mod_M + P[t-1] - P[s-1]) % M == 0
    
    # Part 1: sum(C[r] * (C[r] - 1) // 2 for r in counts)
    # Part 2: sum(C[r] * C[(r - L_mod_M) % M] for r in counts) 
    #         BUT this includes s < t and s = t.
    # Actually, let's just iterate over all r and calculate:
    # For a fixed r_s, the number of t < s such that r_t == (r_s - L_mod_M) % M.
    # This is still loop-like. Let's use the property:
    # Total = sum_{r} (C[r] * C[(r - L_mod_M) % M])
    # This sum counts all pairs (s, t) such that (L_mod_M + P[t-1] - P[s-1]) % M == 0.
    # This is exactly the condition for clockwise distance from s to t to be a multiple of M
    # WHEN s > t.
    # When s < t, the condition is (P[t-1] - P[s-1]) % M == 0.
    
    # Let's use:
    # Ans = sum(C[r] * (C[r] - 1) // 2 for r in counts) # for s < t
    #     + sum(C[r] * C[(r - L_mod_M) % M] for r in counts) # for s > t
    # Wait, the second term counts all (s, t) such that P[t-1] == (P[s-1] - L_mod_M) % M.
    # This includes s < t, s > t, and s = t.
    # Let's use a different approach for s > t:
    # For each s, we want t < s such that P[t-1] % M == (P[s-1] - L_mod_M) % M.
    # This can be solved by iterating through the array and keeping track of counts.
    # Since I can't use loops, I'll use a list comprehension with a helper.
    
    # Let's reconsider:
    # Pair (s, t) is valid if:
    # (s < t AND P[t-1] % M == P[s-1] % M) OR (s > t AND P[t-1] % M == (P[s-1] - L_mod_M) % M)
    
    # Let's use the fact that we can use sum() and list comprehensions.
    # For s < t: the number of pairs is sum(C[r] * (C[r]-1) // 2 for r in counts)
    # For s > t: we need to count pairs (s, t) with t < s and P[t-1] % M == (P[s-1] - L_mod_M) % M.
    # This is equivalent to: for each s, count how many t < s have the required remainder.
    # We can use a trick with a list and a mutable object or a dictionary to track counts 
    # inside a list comprehension, but that's side-effect heavy.
    # Instead, let's use the total count:
    # For a fixed r, there are C[r] indices where P[i] % M == r.
    # For a fixed r', there are C[r'] indices where P[i] % M == r'.
    # If we want to count pairs (s, t) with s > t and P[t-1] % M == r' and P[s-1] % M == r,
    # and we know the indices, we can't easily without loops.
    # BUT, we can use the symmetry:
    # Total pairs (s, t) with s != t such that P[t-1] % M == (P[s-1] - L_mod_M) % M
    # is sum(C[r] * C[(r - L_mod_M) % M]) - (number of s such that P[s-1] == (P[s-1] - L_mod_M) % M).
    # The condition P[s-1] == (P[s-1] - L_mod_M) % M is only true if L_mod_M == 0.
    
    # If L_mod_M == 0:
    # s < t: P[t-1] == P[s-1] (mod M)
    # s > t: P[t-1] == P[s-1] (mod M)
    # Total = sum(C[r] * (C[r] - 1))
    
    # If L_mod_M != 0:
    # s < t: P[t-1] == P[s-1] (mod M) -> sum(C[r] * (C[r] - 1) // 2)
    # s > t: P[t-1] == (P[s-1] - L_mod_M) (mod M)
    # Let r_t = (P[s-1] - L_mod_M) % M.
    # We want to count pairs (s, t) with t < s and P[t-1] % M == r_t.
    # This is hard without loops. Let's use the property:
    # For any two distinct indices i, j, either (i < j) or (i > j).
    # Let r_i = P[i] % M.
    # We are looking for pairs (i, j) such that:
    # (i < j and r_j == r_i)