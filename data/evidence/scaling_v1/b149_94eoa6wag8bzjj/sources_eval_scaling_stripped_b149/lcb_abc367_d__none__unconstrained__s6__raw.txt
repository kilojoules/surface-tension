```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the remaining A_i
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    try:
        n = next(it)
        m = next(it)
        a = list(it)
    except StopIteration:
        return

    # Calculate prefix sums of A_i modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A1%M, (A1+A2)%M, ..., (A1+...+An)%M]
    # We use accumulate to avoid loops
    p = list(accumulate([0] + [x % m for x in a], lambda x, y: (x + y) % m))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # This is 0 mod M if P[t-1] == P[s-1]
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M
    # This is 0 mod M if (P[s-1] - P[t-1]) % M == TotalSum % M
    
    total_sum_mod = p[-1]
    
    # We need to count pairs (s, t) such that:
    # 1. s < t and P[t-1] == P[s-1]
    # 2. s > t and (P[s-1] - P[t-1]) % M == total_sum_mod
    
    # Let counts be the frequency of each remainder in P[0...N-1]
    # Note: P has N+1 elements, but we only care about the positions of the rest areas.
    # Rest area i is at prefix sum P[i-1].
    counts = Counter(p[:-1])
    
    # For a fixed remainder r, there are counts[r] positions.
    # Pairs (s, t) with s < t and P[s-1] == P[t-1] == r:
    # There are counts[r] * (counts[r] - 1) // 2 such pairs.
    # However, the problem asks for pairs (s, t). 
    # If s < t, we check P[t-1] - P[s-1] == 0 mod M.
    # If s > t, we check Total - P[s-1] + P[t-1] == 0 mod M => P[s-1] - P[t-1] == Total mod M.
    
    # Let's redefine:
    # For every pair of indices i, j from {0, ..., N-1} with i < j:
    # Pair (s=i+1, t=j+1) is valid if P[j] - P[i] == 0 mod M
    # Pair (s=j+1, t=i+1) is valid if Total - P[j] + P[i] == 0 mod M
    
    # Let r1 = P[i] and r2 = P[j].
    # Condition 1: r2 - r1 == 0 mod M  => r1 == r2
    # Condition 2: Total - r2 + r1 == 0 mod M => r2 - r1 == Total mod M
    
    # For each remainder r, let c = counts[r].
    # The number of pairs (i, j) with i < j and P[i] == P[j] is c*(c-1)//2.
    # Each such pair contributes 1 to the answer (the s < t case).
    # UNLESS Total mod M == 0, then it also contributes 1 to the s > t case.
    
    # For the s > t case: P[j] - P[i] == Total mod M (with i < j).
    # This is equivalent to P[i] == (P[j] - Total) mod M.
    
    # Let T = total_sum_mod.
    # If T == 0:
    #   Each pair {i, j} with P[i] == P[j] satisfies both s < t and s > t.
    #   Answer = 2 * sum(c * (c-1) // 2) = sum(c * (c-1))
    # If T != 0:
    #   Pairs s < t: P[j] == P[i] => sum(c * (c-1) // 2)
    #   Pairs s > t: P[j] - P[i] == T mod M => sum(counts[r] * counts[(r - T) % m])
    #   Wait, the second sum is over all r, but we need i < j.
    #   Actually, for any two distinct indices i, j, exactly one is smaller.
    #   If P[j] - P[i] == T mod M, then either (i < j and s=j+1, t=i+1) 
    #   or (j < i and s=i+1, t=j+1).
    #   So we just need to count pairs (i, j) such that P[j] - P[i] == T mod M.
    #   This is sum(counts[r] * counts[(r - T) % m]) for all r.
    #   But we must exclude the case where i == j (which would require T == 0).
    
    # Correct Logic:
    # Total Pairs = (Pairs i < j where P[j]-P[i] == 0 mod M) 
    #               + (Pairs i < j where P[j]-P[i] == Total mod M)
    
    # Let C = counts.
    # Part 1: sum(C[r] * (C[r] - 1) // 2 for r in C)
    # Part 2: 
    #   If T == 0: sum(C[r] * (C[r] - 1) // 2 for r in C)
    #   If T != 0: sum(C[r] * C[(r - T) % m] for r in C) 
    #   Wait, the T != 0 case is simpler: for every pair of indices {i, j}, 
    #   one is smaller. If P[larger] - P[smaller] == T mod M, it's a valid (s, t) with s > t.
    #   If P[smaller] - P[larger] == T mod M, it's a valid (s, t) with s < t.
    #   Actually, just:
    #   Ans = sum(C[r] * (C[r] - 1) // 2) + (sum(C[r] * C[(r - T) % m]) if T != 0 else sum(C[r] * (C[r] - 1) // 2))
    
    # Let's refine:
    # For any two distinct indices i, j in {0, ..., N-1}:
    # Let i < j.
    # Pair (s=i+1, t=j+1) is valid if P[j] - P[i] \equiv 0 \pmod M
    # Pair (s=j+1, t=i+1) is valid if Total - (P[j] - P[i]) \equiv 0 \pmod M \Rightarrow P[j] - P[i] \equiv Total \pmod M
    
    # Let T = total_sum_mod.
    # If T == 0:
    #   Both conditions are P[j] - P[i] \equiv 0 \pmod M.
    #   Each pair {i, j} with P[i] == P[j] provides 2 pairs (s, t).
    #   Ans = sum(C[r] * (C[r] - 1))
    # If T != 0:
    #   Condition 1: P[j] == P[i]. Number of pairs = sum(C[r] * (C[r] - 1) // 2)
    #   Condition 2: P[j] - P[i] == T mod M. 
    #   For any two indices i, j, if P[j] - P[i] == T mod M, then exactly one of (i < j) or (j < i) is true.
    #   So we just need to count pairs (i, j) with i != j such that P[j] - P[i] == T mod M.
    #   This is sum(C[r] * C[(r - T) % m]) for all r.
    #   Since T != 0, r != (r - T) % m, so i != j is guaranteed.
    #   However, this counts ordered pairs (i, j). We need i < j.
    #   Actually, for a fixed pair of values {r, (r-T)%m}, the number of ways to pick indices is C[r] * C[(r-T)%m].
    #   For each such choice, one index is smaller. That determines whether it's (s < t) or (s > t).
    #   So we just sum C[r] * C[(r - T) % m] for all r, but that counts each pair once.
    #   Wait, if we sum over all r, we are counting pairs (i, j) such that P[j] = r and P[i] = r-T.
    #   This is exactly what we need.
    
    # Final Formula:
    # If T == 0: Ans = sum(C[r] * (C[r] - 1))
    # If T != 0: Ans = sum(C[r] * (C[r] - 1) // 2) + sum(C[r] * C[(r - T) % m] for r in C)
    # Wait, the second term in T != 0:
    # We want pairs (i, j) with i < j such that P[j] - P[i] == T mod M.
    # This is NOT sum(C[r] * C[(r-T)%m]). That counts all pairs (i, j) regardless of order.
    # Let's use the property: for any two indices i, j, if P[j] - P[i] == T mod M, 
    # then either (i < j) or (j < i).
    # If i < j, then (s=j+1, t=i+1) is a pair.
    # If j < i, then (s=i+1, t=j+1) is a pair.
    # So we just need to count pairs {i, j} such that P[j] - P[i] == T mod M or P[i] - P[j] == T mod M.
    # That is simply sum(C[r] * C[(r - T) % m]) for all r, but divided by... nothing.
    # Let's check: if C[1]=1, C[2]=1, T=1. r=2 => C[2]*C[1] = 1. Pair is {i, j}.
    # One is smaller, one is larger. Exactly one (s, t) pair is formed.
    # So sum(C[r] * C[(r - T) % m]) is correct.
    
    # Let's double check Sample 1: N=4, M=3, A=[2, 1, 4, 3]
    # P = [0, 2, 3, 7, 10] mod 3 = [0, 2, 0, 1, 1]
    # P[:-1] = [0, 2, 0, 1]. C = {0: 2, 2: 1, 1: 1}. T = 1.
    # T != 0:
    # Part 1: 2*1//2 + 1*0//2 + 1*0//2 = 1
    # Part 2: r=0: C[0]*C[-1%3]=C[0]*C[2]=2*1=2; r=2: C[2]*C[1]=1*1=1; r=1: C[1]*C[0]=1*2=2.
    # Total Part 2 = 2 + 1 + 2 = 5? No, that's wrong.
    # The condition for s > t is P[j] - P[i] == T mod M where i < j.
    # Let's re-evaluate:
    # s < t: