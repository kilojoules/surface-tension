The constraint to avoid explicit loops forces a functional programming style, which encourages the use of immutable transformations and high-level abstractions rather than manual state management. For this problem, I will use `itertools.accumulate` to calculate the prefix sums of the distances, and `collections.Counter` to efficiently count the occurrences of each remainder modulo $M$. The final answer is derived using the formula for combinations (specifically, counting pairs with the same remainder) across the prefix sum array.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and map to integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use a trick to unpack the iterator
    # We can't use a for loop, but we can use map/list/next
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Prefix sums: P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A[0], A[0]+A[1], ...]
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # accumulate returns an iterator, we convert to list
    P = list(accumulate(A, lambda x, y: (x + y) % M))
    
    # Total sum of A modulo M
    total_sum = P[-1]
    
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # and (total_sum - P[s-1] + P[t-1]) % M == 0 for s > t
    
    # Let's define S as the set of prefix sums including 0
    # S = [0, P[0], P[1], ..., P[N-1]]
    S = [0] + P
    
    # Count frequencies of each remainder modulo M
    counts = Counter(S)
    
    # For a fixed s and t (s != t):
    # If s < t: we need S[t] - S[s] ≡ 0 (mod M)  => S[t] ≡ S[s] (mod M)
    # If s > t: we need total_sum - S[s] + S[t] ≡ 0 (mod M) => S[s] - S[t] ≡ total_sum (mod M)
    
    # Part 1: s < t
    # For each remainder r, if it appears C times, there are C*(C-1)//2 pairs
    # We use a list comprehension and sum to avoid loops
    ans_st = sum([C * (C - 1) // 2 for C in counts.values()])
    
    # Part 2: s > t
    # We need S[s] - S[t] ≡ total_sum (mod M)
    # For each S[t] = r, we need S[s] = (r + total_sum) % M
    # To avoid double counting and handle s > t, we iterate through the remainders.
    # Let r1 = S[t] and r2 = S[s]. We need r2 - r1 ≡ total_sum (mod M).
    # The number of pairs (t, s) with t < s is counts[r1] * counts[r2].
    # However, we specifically need s > t. 
    # Let's re-evaluate: 
    # For any pair {i, j} with i < j:
    # Clockwise i to j is (S[j] - S[i]) % M
    # Clockwise j to i is (total_sum - (S[j] - S[i])) % M
    
    # Let diff = (S[j] - S[i]) % M.
    # Pair (i, j) is valid if diff == 0.
    # Pair (j, i) is valid if (total_sum - diff) % M == 0.
    
    # If total_sum % M == 0:
    # diff == 0 AND (total_sum - diff) % M == 0 are the same condition.
    # But s != t, so we just count pairs with same remainder.
    # Since total_sum % M == 0, S[N] = S[0] = 0. 
    # The prefix sum array S has N+1 elements.
    # But the rest areas are 1 to N. S[0] corresponds to area 1, S[1] to area 2... S[N-1] to area N.
    # S[N] is just S[0] + total_sum.
    
    # Correct logic:
    # Let R be the remainders of prefix sums for areas 1 to N:
    # R = [0, A[0], A[0]+A[1], ..., A[0]+...+A[N-2]]
    # R has N elements.
    # For any two areas s, t in {1, ..., N} with s < t:
    # Dist(s, t) = (R[t-1] - R[s-1]) % M
    # Dist(t, s) = (total_sum - (R[t-1] - R[s-1])) % M
    
    # Let's redefine R:
    R = S[:-1]
    counts_R = Counter(R)
    
    # Pairs (s, t) with s < t such that Dist(s, t) % M == 0:
    # R[t-1] ≡ R[s-1] (mod M)
    term1 = sum([C * (C - 1) // 2 for C in counts_R.values()])
    
    # Pairs (s, t) with s > t such that Dist(s, t) % M == 0:
    # R[s-1] - R[t-1] ≡ total_sum (mod, M)
    # For each r, we need r_target = (r + total_sum) % M
    # The number of pairs is sum(counts_R[r] * counts_R[r_target])
    # But we must exclude cases where s = t (though s > t already does that).
    # If total_sum % M == 0, then r_target == r, and we get C*C. 
    # But we need s > t, so it's C*(C-1)//2.
    
    # To handle total_sum % M == 0 and others uniformly:
    # If total_sum % M == 0:
    #   s < t: R[t-1] == R[s-1] -> C(C-1)//2
    #   s > t: R[s-1] == R[t-1] -> C(C-1)//2
    #   Total = C(C-1)
    # If total_sum % M != 0:
    #   s < t: R[t-1] == R[s-1] -> C(C-1)//2
    #   s > t: R[s-1] - R[t-1] == total_sum -> counts[r] * counts[(r - total_sum) % M]
    
    # Let's use a simpler approach:
    # For every pair s < t, check Dist(s, t) and Dist(t, s).
    # Dist(s, t) is 0 mod M if R[s-1] == R[t-1].
    # Dist(t, s) is 0 mod M if R[t-1] - R[s-1] == total_sum mod M.
    
    # Total = sum_{r} (counts[r] * counts[(r + total_sum) % M])
    # However, if total_sum % M == 0, this counts s=t.
    # Since s != t, we subtract N if total_sum % M == 0.
    # Wait, if total_sum % M == 0, then R[s-1] == R[t-1] implies both Dist(s,t) and Dist(t,s) are 0 mod M.
    # The number of pairs (s, t) with s != t is:
    # sum_{r} (counts[r] * counts[(r + total_sum) % M]) 
    # minus (N if total_sum % M == 0 else 0)
    
    # Let's check Sample 1: N=4, M=3, A=[2, 1, 4, 3]
    # R = [0, 2, 3, 7] % 3 = [0, 2, 0, 1]
    # total_sum = 10 % 3 = 1
    # counts = {0: 2, 2: 1, 1: 1}
    # r=0: counts[0]*counts[(0+1)%3] = 2 * counts[1] = 2 * 1 = 2
    # r=2: counts[2]*counts[(2+1)%3] = 1 * counts[0] = 1 * 2 = 2
    # r=1: counts[1]*counts[(1+1)%3] = 1 * counts[2] = 1 * 1 = 1
    # Total = 2 + 2 + 1 = 5? Sample says 4.
    # Let's re-read: "minimum number of steps to walk clockwise".
    # s=1, t=3: R[2]-R[0] = 3-0 = 3 (0 mod 3) - OK
    # s=3, t=2: total - (R[2]-R[1]) = 10 - (3-2) = 9 (0 mod 3) - OK
    # s=4, t=1: total - (R[3]-R[0]) = 10 - 7 = 3 (0 mod 3) - OK
    # s=4, t=3: total - (R[3]-R[2]) = 10 - (7-3) = 6 (0 mod 3) - OK
    # My manual trace:
    # R = [0, 2, 0, 1], total = 1
    # s < t: R[t-1] - R[s-1] = 0 mod 3 => (0, 2) since R[0]=R[2]=0. (1 pair)
    # s > t: R[s-1] - R[t-1] = total mod 3 => R[s-1] - R[t-1] = 1 mod 3.
    # Pairs (s, t) with s > t:
    # (2, 1): 2-0 = 2
    # (3, 1): 0-0 = 0
    # (3, 2): 0-2 = -2 = 1 mod 3 (OK)
    # (4, 1): 1-0 = 1 mod 3 (OK)
    # (4, 2): 1-2 = -1 = 2 mod 3
    # (4, 3): 1-0 = 1 mod 3 (OK)
    # Total = 1 + 3 = 4.
    
    # My formula: sum_{r} (counts[r] * counts[(r + total_sum) % M])
    # For Sample 1: R=[0, 2, 0, 1], total=1
    # r=0: counts[0]*counts[1] = 2 * 1 = 2
    # r=1: counts[1]*counts[2] = 1 * 1 = 1
    # r=2: counts[2]*counts[0] = 1 * 2 = 2
    # Total = 5. Why?
    # Because I'm counting (s, t) where s can be anything.
    # Let's use the logic:
    # For a fixed s, we need t such that:
    # 1. t > s and R[t-1] - R[s-1] ≡ 0 (mod M)
    # 2. t < s and R[s-1] - R[t-1] ≡ total_sum (mod M)
    
    # Let's use the property:
    # Pair (s, t) is valid if:
    # (s < t AND R[t-1] ≡ R[s-1]) OR (s > t AND R[s-1] - R[t-1] ≡