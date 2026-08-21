The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm. While not always "cleaner" for complex algorithms, it encourages the use of high-level abstractions and built-in optimizations. To solve this problem, I will use `itertools.accumulate` for prefix sums, `collections.Counter` for frequency mapping, and generator expressions within `sum()` to calculate the final answer.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read N and M from the first line
    # Read A_i values from the remaining input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2
    # ...
    # P_N = A_1 + ... + A_{N-1}
    # Total distance L = A_1 + ... + A_N
    
    # We use accumulate to get prefix sums: [A_1, A_1+A_2, ..., A_1+...+A_N]
    # To get P_1=0, we prepend 0 and take the first N elements.
    prefix_sums = list(accumulate(A))
    P = [0] + prefix_sums[:-1]
    L = prefix_sums[-1]
    
    # The distance from s to t (s < t) is (P_t - P_s).
    # The distance from s to t (s > t) is (L - P_s + P_t).
    # We want distance % M == 0.
    
    # Case 1: s < t
    # (P_t - P_s) % M == 0  =>  P_t % M == P_s % M
    # Case 2: s > t
    # (L + P_t - P_s) % M == 0  =>  P_s % M == (L + P_t) % M
    
    # Let R_i = P_i % M.
    # We need pairs (s, t) such that:
    # 1. s < t and R_s == R_t
    # 2. s > t and R_s == (L + R_t) % M
    
    # Let count[r] be the number of i in {1...N} such that R_i = r.
    # For a fixed r, the number of pairs (s, t) with s < t and R_s = R_t = r
    # is count[r] * (count[r] - 1) // 2.
    # Total for Case 1: sum(count[r] * (count[r] - 1) // 2 for r in range(M))
    
    # For Case 2, we need R_s = (L + R_t) % M with s > t.
    # This is trickier because of the s > t constraint.
    # However, we can observe that the total number of pairs (s, t) with s != t
    # such that dist(s, t) % M == 0 is what we need.
    
    # Let's use a different approach:
    # For every s, we want to find t != s such that dist(s, t) % M == 0.
    # dist(s, t) = (P_t - P_s) mod L (effectively).
    # Specifically, clockwise distance is:
    # If s < t: P_t - P_s
    # If s > t: L - (P_s - P_t) = L + P_t - P_s
    
    # In both cases, we want (P_t - P_s + (L if s > t else 0)) % M == 0.
    # This is equivalent to:
    # If s < t: P_t % M == P_s % M
    # If s > t: P_s % M == (L + P_t) % M
    
    # Let's use the property:
    # Total pairs = (Pairs s < t where P_t % M == P_s % M) 
    #               + (Pairs s > t where P_s % M == (L + P_t) % M)
    
    # Let R = [p % M for p in P]
    # Let counts = Counter(R)
    # Case 1 (s < t): sum(v * (v - 1) // 2 for v in counts.values())
    
    # Case 2 (s > t): 
    # We need R_s = (L + R_t) % M.
    # Let target_R_t = (R_s - L) % M.
    # For a fixed s, we need the number of t < s such that R_t = (R_s - L) % M.
    # This can be solved by iterating through the list and keeping track of counts.
    # But since we can't use loops, we can use a different logic:
    # The total number of pairs (s, t) with s != t such that dist(s, t) % M == 0 is:
    # For each s, we seek t such that:
    # 1. t > s and P_t % M = P_s % M
    # 2. t < s and P_t % M = (P_s - L) % M
    
    # Let's simplify:
    # We want (P_t - P_s) % M == 0 if s < t
    # We want (P_t - P_s + L) % M == 0 if s > t
    
    # Let R_i = P_i % M.
    # We want:
    # s < t: R_t = R_s
    # s > t: R_t = (R_s - L) % M
    
    # Let's use a frequency map of R.
    # For a fixed R_s = r:
    # Number of t > s with R_t = r is (count[r] - (number of i <= s with R_i = r))
    # Number of t < s with R_t = (r - L) % M is (number of i < s with R_i = (r - L) % M)
    
    # Total = sum_{s=1 to N} [ (count[R_s] - count_upto[s][R_s]) + count_upto[s-1][(R_s - L) % M] ]
    # This still looks like a loop. Let's use the fact that:
    # Sum_{s < t} [R_s == R_t] = sum(v*(v-1)//2 for v in counts.values())
    # Sum_{s > t} [R_t == (R_s - L) % M] = Sum_{t < s} [R_t == (R_s - L) % M]
    
    # Let L_mod = L % M.
    # We want pairs (s, t) with s < t such that:
    # (R_t - R_s) % M == 0  OR  (R_t - R_s + L_mod) % M == 0
    # Wait, the second condition is for s > t.
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # 1. s < t AND R_s == R_t
    # 2. s > t AND R_t == (R_s - L_mod) % M
    
    # Let's use the symmetry. 
    # Sum_{s < t} [R_s == R_t] is easy.
    # For the second part: Sum_{s > t} [R_t == (R_s - L_mod) % M]
    # This is Sum_{t < s} [R_t == (R_s - L_mod) % M]
    # This is Sum_{t < s} [R_s == (R_t + L_mod) % M]
    
    # Let's use a list of indices for each remainder:
    # indices = {r: [i for i, val in enumerate(R) if val == r]}
    # For a fixed t, we need s > t such that R_s = (R_t + L_mod) % M.
    # The number of such s is (count[(R_t + L_mod) % M] - (number of i <= t with R_i = (R_t + L_mod) % M))
    
    # Actually, the simplest way to calculate Sum_{t < s} [R_s == (R_t + L_mod) % M]:
    # It is Sum_{t=1 to N} (count[(R_t + L_mod) % M] - (number of i <= t with R_i = (R_t + L_mod) % M))
    # = Sum_{t=1 to N} count[(R_t + L_mod) % M] - Sum_{t=1 to N} (number of i <= t with R_i = (R_t + L_mod) % M)
    
    # Let's use the property:
    # Total = Sum_{s < t} [R_s == R_t] + Sum_{t < s} [R_s == (R_t + L_mod) % M]
    # The first term is sum(v*(v-1)//2 for v in counts.values())
    # For the second term, if L_mod == 0, it's the same as the first term.
    # If L_mod != 0, then R_s and R_t must be different.
    # Sum_{t < s} [R_s == (R_t + L_mod) % M] 
    # Since R_s != R_t, the condition t < s is just a way of counting pairs.
    # For any pair {t, s} with R_s == (R_t + L_mod) % M, either (t < s) or (s < t).
    # If L_mod != 0, then R_s != R_t, so we can't have both.
    # Wait, if L_mod != 0, then R_s = (R_t + L_mod) % M and R_t = (R_s + L_mod) % M 
    # cannot both be true unless 2*L_mod == 0 % M.
    
    # Let's use a different approach for the second term:
    # Sum_{t < s} [R_s == (R_t + L_mod) % M]
    # Let's use a list of prefix counts.
    # Since we can't use loops, we can use a list comprehension to create a list of 
    # "number of times R_i has appeared so far".
    # But that requires a loop or a mutable state.
    # Let's use the fact that:
    # Sum_{t < s} [R_s == (R_t + L_mod) % M] = 
    # Sum_{r=0 to M-1} (count[r] * count[(r + L_mod) % M]) 
    # MINUS Sum_{t=1 to N} [R_t == (R_t + L_mod) % M and t is the 's' in the pair]
    # Actually:
    # Sum_{t, s} [R_s == (R_t + L_mod) % M] = Sum_{r=0 to M-1} (count[r] * count[(r + L_mod) % M])
    # This sum includes pairs where t < s, t > s, and t = s.
    # t = s is only possible if L_mod == 0.
    # If L_mod == 0, the condition is R_s == R_t, and we want s > t. This is count[r]*(count[r]-1)//2.
    # If L_mod != 0, then R_s != R_t.
    # For any pair {i, j} with i < j, either:
    # 1. R_j = (R_i + L_mod) % M  (This is our Case 2)
    # 2. R_i = (R_j + L_mod) % M  (This is Case 2 if we swapped s and t)
    # 3. Neither.
    # These two are mutually exclusive if L_mod != 0 and 2*L_mod != 0 % M.
    # This is getting confusing. Let