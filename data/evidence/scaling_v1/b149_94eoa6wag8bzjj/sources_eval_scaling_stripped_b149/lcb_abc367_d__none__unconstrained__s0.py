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
    
    # Calculate prefix sums modulo M. 
    # P[i] = sum(A[0...i-1]) % M. P[0] = 0.
    # We use accumulate to avoid loops.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # P now has N+1 elements. We only need P[0]...P[N-1] for the starting points.
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M == 0 => P[t-1] == P[s-1].
    # Note: the problem says distance from i to i+1 is A_i.
    # So dist(1, 2) = A_1, dist(1, 3) = A_1 + A_2, etc.
    # Let's redefine: S_i is the distance from rest area 1 to rest area i.
    # S_1 = 0
    # S_2 = A_1
    # S_3 = A_1 + A_2
    # S_i = sum(A[0...i-2])
    
    # Correct prefix sums for rest areas 1 to N:
    # S = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    S = list(accumulate(A[:-1], lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of all A_i modulo M
    total_sum = sum(A) % M
    
    # Count occurrences of each remainder in S
    counts = Counter(S)
    
    # For s < t: dist(s, t) = (S[t-1] - S[s-1]) % M == 0  => S[t-1] == S[s-1]
    # Number of pairs (s, t) with s < t is sum(c * (c - 1) // 2)
    ans_st = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For s > t: dist(s, t) = (total_sum - S[s-1] + S[t-1]) % M == 0
    # => S[s-1] - S[t-1] == total_sum (mod M)
    # => S[s-1] - total_sum == S[t-1] (mod M)
    # For each s, we need to count t < s such that S[t-1] == (S[s-1] - total_sum) % M.
    # This is equivalent to: for each remainder r, count pairs (r, (r - total_sum) % M).
    # However, we must exclude the case where s = t (already handled by s > t).
    # The total pairs (s, t) with s > t is sum(counts[r] * counts[(r - total_sum) % M])
    # BUT, if total_sum % M == 0, then r == (r - total_sum) % M, 
    # and we are counting pairs (s, t) where S[s-1] == S[t-1].
    # Since we need s > t, if total_sum % M == 0, it's the same as the s < t case.
    # If total_sum % M != 0, then r != (r - total_sum) % M, so all such pairs have s != t.
    
    # Let's use a more direct approach for s > t:
    # We want pairs (s, t) such that 1 <= t < s <= N and (total_sum - S[s-1] + S[t-1]) % M == 0.
    # This is S[t-1] == (S[s-1] - total_sum) % M.
    # Let r_s = S[s-1] and r_t = S[t-1].
    # We need r_t == (r_s - total_sum) % M.
    
    # If total_sum % M == 0:
    # We need r_t == r_s. For each group of size c, there are c*(c-1)//2 pairs.
    # If total_sum % M != 0:
    # We need r_t == (r_s - total_sum) % M.
    # This is sum(counts[r] * counts[(r - total_sum) % M]) 
    # But we only want t < s. 
    # Actually, the condition s > t is symmetric to s < t if we consider the whole circle.
    # Let's just calculate the total pairs (s, t) with s != t.
    # A pair (s, t) is valid if (S[t-1] - S[s-1]) % M == 0 (for s < t)
    # or (total_sum - S[s-1] + S[t-1]) % M == 0 (for s > t).
    
    # Let's use the property:
    # For a fixed s, we want t != s such that dist(s, t) % M == 0.
    # If s < t, we need S[t-1] == S[s-1] (mod M).
    # If s > t, we need S[t-1] == (S[s-1] - total_sum) (mod M).
    
    # Total = sum_{s=1 to N} [ (count of t > s with S[t-1] == S[s-1]) 
    #                      + (count of t < s with S[t-1] == (S[s-1] - total_sum) % M) ]
    
    # Let's evaluate this:
    # Part 1: sum_{s < t} [S[t-1] == S[s-1]] = sum(c*(c-1)//2)
    # Part 2: sum_{s > t} [S[t-1] == (S[s-1] - total_sum) % M]
    # Let target(r) = (r - total_sum) % M.
    # We want to count pairs (s, t) with t < s and S[t-1] == target(S[s-1]).
    # This can be done by iterating s from 1 to N and keeping track of counts of S[t-1] seen so far.
    
    # To avoid loops, we can use the total counts:
    # If total_sum % M == 0:
    #   Part 2 is also sum(c*(c-1)//2).
    # If total_sum % M != 0:
    #   We need to count pairs (s, t) with t < s and S[t-1] == target(S[s-1]).
    #   This is not simply counts[r] * counts[target(r)].
    #   Wait, the problem can be solved by:
    #   Total = sum_{r=0 to M-1} (counts[r] * counts[(r + total_sum) % M])
    #   But we must subtract cases where s=t and dist(s, s) is considered.
    #   The problem says s != t.
    #   If we take all pairs (s, t) and check (dist(s, t) % M == 0):
    #   For a fixed s, t is valid if:
    #   1. t > s and S[t-1] == S[s-1] (mod M)
    #   2. t < s and S[t-1] == (S[s-1] - total_sum) (mod M)
    
    # Let's use the "total" approach:
    # For each s, the number of t != s such that dist(s, t) % M == 0 is:
    # If total_sum % M == 0:
    #    The condition is always S[t-1] == S[s-1] (mod M).
    #    There are counts[S[s-1]] - 1 such t's.
    #    Total = sum(c * (c - 1))
    # If total_sum % M != 0:
    #    If S[t-1] == S[s-1], then dist(s, t) % M == 0 ONLY IF s < t.
    #    If S[t-1] == (S[s-1] - total_sum) % M, then dist(s, t) % M == 0 ONLY IF s > t.
    #    Wait, that's not right.
    #    If S[t-1] == S[s-1], then dist(s, t) = S[t-1] - S[s-1] = 0 (mod M) for s < t.
    #    And dist(s, t) = total_sum + S[t-1] - S[s-1] = total_sum (mod M) for s > t.
    #    If S[t-1] == (S[s-1] - total_sum), then dist(s, t) = S[t-1] - S[s-1] = -total_sum (mod M) for s < t.
    #    And dist(s, t) = total_sum + S[t-1] - S[s-1] = 0 (mod M) for s > t.
    
    # So if total_sum % M != 0:
    # For a fixed s, t is valid if:
    # (s < t AND S[t-1] == S[s-1]) OR (s > t AND S[t-1] == (S[s-1] - total_sum) % M)
    # Total = sum_{r} [ (count of r) * (count of r after s) + (count of target(r)) * (count of target(r) before s) ]
    # This is equivalent to:
    # sum_{r} [ (count of r) * (count of r) - (count of r) ] / 2  <-- this is for s < t
    # + sum_{r} [ (count of r) * (count of target(r)) ] <-- this is for s > t, but only if we don't double count.
    # Let's be precise:
    # Total = sum_{s < t} [S[s-1] == S[t-1]] + sum_{s > t} [S[t-1] == (S[s-1] - total_sum) % M]
    # Part 1 = sum(c * (c - 1) // 2)
    # Part 2 = sum_{s=1 to N} (count of t < s such that S[t-1] == (S[s-1] - total_sum) % M)
    
    # To calculate Part 2 without loops:
    # Let f(r) = (r - total_sum) % M.
    # We want sum_{s=1 to N} count(t < s, S[t-1] == f(S[s-1])).
    # This is sum_{r} sum_{s: S[s-1]=r} count(t < s, S[t-1] == f(r)).
    # If r == f(r) (which happens if total_sum % M == 0), this is sum(c*(c-1)//2).
    # If r != f(r), this is sum_{s: S[s-1]=r} count(t < s, S[t-1] == f(r)).
    # This depends on the relative order of indices.
    # Let's use the property: 
    # sum_{s > t} [S[t-1] == f(S[s-1])] + sum_{s < t} [S[t-1] == f(S[s-1])] = counts[r] * counts[f(r)]
    # And sum_{s < t} [S[t-1] == f(S[