The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state changes. This approach leverages Python's powerful built-in higher-order functions and sequence expressions, which are often more concise and can be more efficient for data processing tasks. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions for mapping values to their remainders modulo $M$, and `collections.Counter` to efficiently count occurrences of these remainders. The final answer is derived by iterating over the frequency distribution of remainders and calculating the number of pairs $(s, t)$ that satisfy the condition using the property that the distance between two points is a multiple of $M$ if their prefix sums have the same remainder modulo $M$.

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
    
    # Prefix sums of distances: P[i] is distance from rest area 1 to rest area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN-1]
    # We use accumulate to get prefix sums and then take modulo M
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # P[0] = 0, P[1] = A1 % M, P[2] = (A1 + A2) % M, ...
    # We only need the first N prefix sums (from area 1 to N)
    # The distance from area 1 to area 1 is 0, but s != t.
    # Let S_i be the distance from area 1 to area i.
    # S_1 = 0
    # S_2 = A_1
    # S_3 = A_1 + A_2 ...
    # S_N = A_1 + ... + A_{N-1}
    # Total = A_1 + ... + A_N
    
    prefix_sums = list(accumulate(A, lambda x, y: x + y))
    total_sum = prefix_sums[-1]
    
    # We are interested in S_i % M for i = 1...N
    # S_1 = 0
    # S_i = prefix_sums[i-2] for i = 2...N
    s_vals = [0] + [prefix_sums[i] for i in range(N-1)]
    remainders = [v % M for v in s_vals]
    
    # Count occurrences of each remainder
    counts = Counter(remainders)
    
    # For a fixed s and t (s < t):
    # Distance is (S_t - S_s) % M == 0  => S_t % M == S_s % M
    # For a fixed s and t (s > t):
    # Distance is (Total - S_s + S_t) % M == 0 => (S_s - S_t) % M == Total % M
    
    # Case 1: s < t
    # For each remainder r, if there are c copies, there are c*(c-1)//2 pairs
    ans_st = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need (S_s - S_t) % M == Total % M
    # Let R = Total % M. We need S_s % M - S_t % M == R (mod M)
    # S_t % M == (S_s % M - R) % M
    R = total_sum % M
    
    # For each s, we need to count t < s such that S_t % M == (S_s % M - R) % M
    # However, the problem asks for pairs (s, t). 
    # Let's re-evaluate: 
    # Clockwise distance from s to t:
    # If s < t: Dist = S_t - S_s
    # If s > t: Dist = (Total - S_s) + S_t
    
    # We want Dist % M == 0.
    # If s < t: S_t % M == S_s % M
    # If s > t: S_t % M == (S_s - Total) % M
    
    # Let's use the counts dictionary to calculate this without loops.
    # For s < t, we already have ans_st.
    # For s > t, for every remainder r1 (representing S_s), 
    # we need remainder r2 (representing S_t) such that r2 == (r1 - R) % M.
    # The number of such pairs is sum(counts[r1] * counts[(r1 - R) % M])
    # But we must exclude the case where s == t.
    # If s == t, the distance is not defined by the problem (s != t), 
    # but the formula (Total - S_s) + S_s = Total.
    # So if Total % M == 0, the case s == t would be counted.
    
    # Correct logic for s > t:
    # For every pair (s, t) with s > t, we check if (Total - S_s + S_t) % M == 0.
    # This is equivalent to S_t % M == (S_s - Total) % M.
    # Let r1 = S_s % M and r2 = S_t % M.
    # We need r2 == (r1 - R) % M.
    # Total pairs (s, t) with s > t is harder with just Counter.
    # Let's use the property:
    # Total pairs = Sum_{r1} (count[r1] * count[(r1 - R) % M])
    # This sum includes cases where s < t, s > t, and s == t.
    # Wait, the logic above is simpler:
    # For a fixed s and t, the clockwise distance is:
    # If s < t: S_t - S_s
    # If s > t: Total - S_s + S_t
    
    # Let's use the remainders list and a frequency map.
    # For s < t: we need S_t % M == S_s % M.
    # For s > t: we need S_t % M == (S_s - Total) % M.
    
    # Let's calculate:
    # 1. Pairs (s, t) with s < t and S_s % M == S_t % M
    # 2. Pairs (s, t) with s > t and S_t % M == (S_s - Total) % M
    
    # For 1: sum(c * (c-1) // 2 for c in counts.values())
    # For 2: 
    # We need to count pairs (s, t) such that s > t and S_t % M == (S_s - R) % M.
    # This is tricky without a loop. Let's use the fact that:
    # Total pairs (s, t) with s != t such that Dist % M == 0 is:
    # Sum_{s=1 to N} (count of t != s such that Dist(s, t) % M == 0)
    
    # For a fixed s, t is determined by the remainder.
    # If t > s, we need S_t % M == S_s % M.
    # If t < s, we need S_t % M == (S_s - R) % M.
    
    # Let's use a different approach:
    # For each remainder r, let C(r) be the number of times it appears in S_1...S_N.
    # The number of pairs (s, t) with s < t and S_s % M == S_t % M is C(r)*(C(r)-1)//2.
    # The number of pairs (s, t) with s > t and S_t % M == (S_s - R) % M is:
    # This is harder because of the s > t constraint.
    # Actually, let's just iterate over the remainders:
    # For a fixed s, the number of t > s such that S_t % M == S_s % M is (count of S_t % M == r) - (count of S_i % M == r for i <= s).
    # This is still a loop.
    
    # Wait! The constraint to avoid loops applies to the final implementation.
    # I can use map/reduce/comprehensions.
    # Let's use the property:
    # Total = Sum_{s=1 to N} [ (count of t > s with S_t % M == S_s % M) + (count of t < s with S_t % M == (S_s - R) % M) ]
    
    # Let's use a list to store the running count of remainders.
    # But we can't use a loop to build the running count.
    # We can use a list comprehension with a mutable object (like a dictionary) to track state.
    # However, that's essentially a loop.
    # Let's use the mathematical property:
    # Total pairs = Sum_{r} (C(r) * C((r - R) % M))
    # This sum counts all pairs (s, t) such that:
    # If s < t, then S_t % M == S_s % M AND S_t % M == (S_s - R) % M  => R % M == 0
    # If s > t, then S_t % M == (S_s - R) % M
    # If s == t, then S_s % M == (S_s - R) % M => R % M == 0
    
    # Let's refine:
    # We want pairs (s, t) with s != t.
    # If R % M != 0:
    #   s < t: S_t % M == S_s % M
    #   s > t: S_t % M == (S_s - R) % M
    #   These two conditions are mutually exclusive because S_t % M cannot be both S_s % M and (S_s - R) % M.
    #   So we can just sum C(r) * C(r) is not right.
    #   For s < t, we have sum(C(r)*(C(r)-1)//2).
    #   For s > t, we need to count pairs (s, t) with s > t and S_t % M == (S_s - R) % M.
    #   This is equivalent to: for each s, count t < s with S_t % M == (S_s - R) % M.
    
    # Let's use the fact that we can use `sum` and list comprehensions.
    # To avoid explicit loops, we can use a helper function with `functools.reduce`.
    from functools import reduce
    
    # state = (total_count, current_counts_dict)
    # We process S_i one by one.
    def accumulate_pairs(state, s_val):
        total, counts_dict = state
        r_s = s_val % M
        r_t_for_gt = (r_s - R) % M
        # Pairs with t < s: S_t % M == r_t_for_gt
        # Pairs with t > s: will be handled when we encounter those t's
        # Actually, let's just count t < s for both conditions:
        # 1. t < s and S_t % M == S_s % M  (This is the s > t case for the first condition)
        # 2. t < s and S_t % M == (S_s - R) % M (This is the s > t case for the second condition)
        # Wait, the conditions were:
        # s < t: S_t % M == S_s % M
        # s > t: S_t % M == (S_s - R) % M
        # So for a fixed s, we need to count t < s such that S_t % M == (S_s - R) % M.
        # And for the s < t case, we can just use the C(r)*(C(r)-1)//2 formula.
        
        new_counts = counts_dict.copy()
        new_counts[r_