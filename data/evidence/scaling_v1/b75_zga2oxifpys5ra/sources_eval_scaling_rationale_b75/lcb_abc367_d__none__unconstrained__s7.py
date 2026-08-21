The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state changes. In idiomatic Python, this encourages the use of built-in higher-order functions and comprehensions which are often more concise and can be more performant. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions for mapping values to their remainders modulo $M$, and `collections.Counter` to efficiently count occurrences of these remainders. The final answer is derived by iterating over the frequency map and calculating combinations of pairs that satisfy the modular arithmetic condition.

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
    
    # Calculate prefix sums of distances
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN-1]
    P = list(accumulate(A, initial=0))
    
    # We are interested in distances modulo M
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (P[N] - P[s-1]) + P[t-1]
    
    # Let R[i] = P[i] % M
    R = [p % M for p in P[:N]]
    counts = Counter(R)
    
    # Total distance around the lake
    total_dist_mod = sum(A) % M
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => R[t-1] == R[s-1]
    # For a pair (s, t) with s > t:
    # Distance is (total_dist_mod + R[t-1] - R[s-1]) % M == 0 
    # => R[s-1] == (total_dist_mod + R[t-1]) % M
    
    # Case 1: s < t
    # For each remainder r, if there are c copies, there are c*(c-1)//2 pairs
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need R[s-1] = (total_dist_mod + R[t-1]) % M
    # Let r_t = R[t-1]. We need count of r_s = (total_dist_mod + r_t) % M
    # We sum over all possible remainders r in the Counter
    ans_s_gt_t = sum(
        counts[r] * counts[(total_dist_mod + r) % M]
        for r in counts
    )
    
    # Special handling for s > t: 
    # If total_dist_mod == 0, then R[s-1] == R[t-1].
    # The logic above counts all pairs (s, t) where R[s-1] == R[t-1].
    # However, we must ensure s != t.
    # If total_dist_mod == 0, the sum counts pairs where s=t, which we must subtract.
    # Also, the logic for s > t and s < t becomes identical if total_dist_mod == 0.
    
    # Correct logic for s > t:
    # For a fixed t, we need s > t such that R[s-1] = (total_dist_mod + R[t-1]) % M.
    # This is tricky without loops. Let's refine:
    # Total pairs (s, t) such that dist(s, t) % M == 0:
    # 1. s < t: R[t-1] - R[s-1] = 0 mod M  => R[t-1] = R[s-1]
    # 2. s > t: total + R[t-1] - R[s-1] = 0 mod M => R[s-1] = (total + R[t-1]) mod M
    
    # Let's use a different approach for the sum to avoid loops and recursion:
    # For s < t, we need pairs (i, j) with 0 <= i < j < N and R[i] == R[j].
    # For s > t, we need pairs (i, j) with 0 <= j < i < N and R[i] == (total + R[j]) % M.
    
    # The number of pairs (i, j) with i < j and R[i] == R[j] is sum(c*(c-1)//2)
    # The number of pairs (i, j) with i > j and R[i] == (total + R[j]) % M:
    # This is harder because the condition depends on the relative order.
    # Actually, the problem says "minimum number of steps to walk clockwise".
    # From s to t (s < t), distance is P[t-1] - P[s-1].
    # From s to t (s > t), distance is (P[N] - P[s-1]) + P[t-1].
    
    # Let's use the property:
    # Total = sum_{s=1 to N} sum_{t=1, t!=s to N} [dist(s, t) % M == 0]
    # For a fixed s, we need t != s such that:
    # If t > s: P[t-1] % M == P[s-1] % M
    # If t < s: (P[N] + P[t-1]) % M == P[s-1] % M
    
    # Let R[i] = P[i] % M for i = 0...N-1
    # For each i in 0...N-1:
    # Count j > i such that R[j] == R[i]
    # Count j < i such that (total_dist_mod + R[j]) % M == R[i]
    
    # This can be solved by iterating through the array and maintaining a running count.
    # Since loops are forbidden, we can use a custom function with reduce or a clever comprehension.
    # However, the most reliable way to count pairs without a loop is to use the global counts
    # and handle the s < t and s > t cases separately.
    
    # For s < t: we need R[s-1] == R[t-1]. Total is sum(c*(c-1)//2).
    # For s > t: we need R[s-1] == (total_dist_mod + R[t-1]) % M.
    # Let's denote R[s-1] as r_s and R[t-1] as r_t.
    # We need r_s == (total_dist_mod + r_t) % M with s > t.
    # This is equivalent to saying for every pair (r_s, r_t) that satisfies the condition,
    # we count how many such pairs exist in the sequence.
    # But the condition s > t is specific.
    
    # Wait, the total distance is the same for all pairs (s, t) if we consider the 
    # relative positions. Let's use the property:
    # Total pairs = sum_{r=0 to M-1} (count[r] * count[(r + total_dist_mod) % M])
    # If total_dist_mod == 0, this counts pairs where r_s == r_t, including s=t.
    # There are N such pairs where s=t. So we subtract N.
    # If total_dist_mod != 0, then r_s != r_t, so s=t is never counted.
    
    # Let's verify:
    # If total_dist_mod == 0:
    # s < t: R[s-1] == R[t-1]
    # s > t: R[s-1] == R[t-1]
    # Total = 2 * sum(c*(c-1)//2) = sum(c*(c-1))
    # Our formula: sum(c * c) - N = sum(c^2) - sum(c) = sum(c(c-1)). Correct.
    
    # If total_dist_mod != 0:
    # s < t: R[s-1] == R[t-1]
    # s > t: R[s-1] == (total_dist_mod + R[t-1]) % M
    # These two conditions are mutually exclusive because total_dist_mod != 0.
    # Total = sum(c*(c-1)//2) + sum_{r} (count[r] * count[(total_dist_mod + r) % M])
    # Wait, the second term is for s > t. For a fixed r_t, we need r_s = (total + r_t) % M.
    # Since we need s > t, we can't just multiply total counts.
    # Actually, we can! For any two indices i, j with R[i] = (total + R[j]) % M,
    # either i > j or i < j.
    # If i > j, it's a valid pair (s=i+1, t=j+1).
    # If i < j, then R[j] = (total + R[i]) % M is NOT necessarily true.
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # 1. s < t and R[t-1] = R[s-1]
    # 2. s > t and R[s-1] = (total_dist_mod + R[t-1]) % M
    
    # Let's use the symmetry.
    # Let f(r) be the count of remainder r.
    # Total = sum_{r} [f(r)*(f(r)-1)//2] + sum_{i < j} [R[i] == (total_dist_mod + R[j]) % M]
    # The second term is the number of pairs (i, j) with i < j such that R[i] - R[j] == total_dist_mod % M.
    # This can be solved by iterating and keeping track of counts.
    # Since we can't use loops, we can use a generator expression inside sum() and a mutable object (like a list) to track state.
    
    state = [0] * M
    # We want to count pairs (i, j) with i < j such that R[i] == (total_dist_mod + R[j]) % M
    # As we iterate j from 0 to N-1, we need to know how many i < j satisfy R[i] == (total_dist_mod + R[j]) % M.
    # We can use a list comprehension to update the state and return the value.
    
    # To avoid loops, we use a helper function and map/reduce or a list comprehension.
    # But we need to maintain state. A list can be mutated inside a list comprehension.
    
    def count_s_gt_t(r_val):
        target = (total_dist_mod + r_val) % M
        # This is for s > t, so we need R[s-1] == (total + R[t-1]) % M
        # Here r_val is R[t-1], we need to count how many R[s-1] appear after it.
        # That's hard. Let's count how many R[s-1] appeared BEFORE it.
        # If we iterate i from 0 to N-1, and i is our 's', we need R[i] == (total + R[j]) % M for j < i.
        # So for a fixed i, we need R[j] == (R[i] - total_dist_mod) % M.
        res = state[(R[i] - total_dist_mod) % M] if 'i' in locals() else 0 # This won't work
        return res

    # Correct approach using a list comprehension to simulate a loop:
    # We use a list to store counts and a list comprehension to aggregate the results.
    # The expression `(state.__setitem__(slice(None), state), state[target])` is not quite right.
    # We can use a function that updates the state and returns the current count.
    
    def update_and_get(r_