The constraint to avoid explicit loops forces a functional programming style, which encourages the use of high-level abstractions and built-in optimizations rather than manual index management. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, `map` to apply the modulo operator, and `collections.Counter` to count the occurrences of each remainder. The final answer is derived by calculating the number of pairs for each remainder using the formula $n(n-1)/2$ and summing them up.

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
    # The distance from s to t (s < t) is (P_t - P_s).
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t).
    # Both conditions simplify to: (P_t - P_s) % M == 0 
    # if we consider the prefix sums modulo M.
    # Wait, the distance from s to t is a multiple of M if:
    # (P_t - P_s) % M == 0 for s < t
    # (Total_Sum + P_t - P_s) % M == 0 for s > t
    
    # Let's redefine: P_i is the clockwise distance from area 1 to area i.
    # P_1 = 0
    # P_i = sum(A_1 ... A_{i-1})
    # Distance s -> t (s < t) is P_t - P_s.
    # Distance s -> t (s > t) is (P_N + A_N) - P_s + P_t.
    
    # Let S = sum(A_1 ... A_N).
    # We want (P_t - P_s) % M == 0 for s < t
    # AND (S + P_t - P_s) % M == 0 for s > t.
    # This is not a simple counting problem because the condition changes based on s < t.
    # Actually, the problem asks for pairs (s, t) where s != t.
    # Let X_i = P_i % M.
    # For s < t: (X_t - X_s) % M == 0  => X_t == X_s
    # For s > t: (S + X_t - X_s) % M == 0 => X_s - X_t == S % M
    
    # Let's use the prefix sums:
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # Total S = A_1 + ... + A_N
    
    # Correct approach:
    # Let P_i be the distance from area 1 to area i.
    # P_1 = 0, P_2 = A_1, ..., P_N = A_1 + ... + A_{N-1}
    # Distance s -> t (s < t): P_t - P_s
    # Distance s -> t (s > t): (S - P_s) + P_t
    
    # We need:
    # 1. s < t and (P_t - P_s) % M == 0
    # 2. s > t and (S + P_t - P_s) % M == 0
    
    # Let X_i = P_i % M.
    # Condition 1: X_t == X_s (for s < t)
    # Condition 2: X_s - X_t == S % M (for s > t)
    
    # Let counts be the frequency of each remainder X_i.
    # Total pairs (s, t) with s < t and X_s == X_t is sum(count * (count - 1) // 2).
    # Total pairs (s, t) with s > t and X_s - X_t == S % M:
    # This is sum(count(X_t + S % M) * count(X_t)) but only for s > t.
    # This is tricky. Let's use the property that we can iterate through the array.
    
    # Let's use a different approach:
    # For a fixed t, we need s < t such that X_s = X_t
    # AND s > t such that X_s = (X_t + S) % M.
    
    # Let's use a loop-free way to calculate this:
    # 1. Calculate P_i % M for all i.
    # 2. Use a Counter to get frequencies of all X_i.
    # 3. The number of pairs (s, t) with s < t and X_s = X_t is 
    #    the sum of (c * (c - 1) // 2) for each count c in Counter.
    # 4. The number of pairs (s, t) with s > t and X_s - X_t = S % M is
    #    the sum of (count(X_t) * count((X_t + S) % M)) 
    #    BUT we must subtract the cases where s < t.
    #    Wait, the condition s > t is strict.
    #    Let's use the fact that:
    #    Total pairs (s, t) with s != t such that dist(s, t) % M == 0 is:
    #    Sum_{t=1 to N} [ (count of s < t where X_s = X_t) + (count of s > t where X_s = (X_t + S) % M) ]
    
    # Let's simplify:
    # For every pair {i, j} with i < j:
    # Pair (i, j) is valid if X_j - X_i = 0 (mod M)
    # Pair (j, i) is valid if S + X_i - X_j = 0 (mod M) => X_j - X_i = S (mod M)
    
    # Let C be the Counter of X_i.
    # Total = Sum_{val} (C[val] * (C[val] - 1) // 2)  <-- for s < t
    # Total += Sum_{val} (C[val] * C[(val + S) % M])  <-- for s > t
    # Wait, the second term counts all pairs (s, t) such that X_s - X_t = S % M.
    # This includes both s < t and s > t.
    # But we only want s > t.
    # Let's use the property:
    # For a fixed pair {i, j} with i < j:
    # It contributes to the answer if X_i == X_j (as (i, j))
    # It contributes to the answer if X_j - X_i == S % M (as (j, i))
    
    # So the answer is:
    # Sum_{i < j} [I(X_i == X_j) + I(X_j - X_i == S % M)]
    # = Sum_{i < j} I(X_i == X_j) + Sum_{i < j} I(X_j - X_i == S % M)
    
    # The first term is Sum (C[v] * (C[v]-1) // 2).
    # The second term: Sum_{i < j} I(X_j - X_i == S % M).
    # This can be solved by iterating through the array and keeping track of counts.
    # Since I can't use loops, I'll use a custom function with reduce or a list comprehension 
    # that simulates a loop, but the prompt says "avoid explicit loops" (for/while).
    # I can use a recursive function or map/reduce. However, the most "Pythonic" 
    # way to avoid loops while maintaining state is using a generator or reduce.
    
    # Actually, I can use a list comprehension to build a list of counts and then sum them.
    # To calculate Sum_{i < j} I(X_j - X_i == S % M):
    # We can use a dictionary to store counts of X_i seen so far.
    # Since I cannot use a loop, I will use `functools.reduce`.
    
    from functools import reduce

    # P_i values
    P = list(accumulate([0] + A)) # P_1 to P_{N+1}
    # We only need P_1 to P_N
    X = [p % M for p in P[:N]]
    S_mod = sum(A) % M
    
    # Part 1: s < t and X_s == X_t
    counts = Counter(X)
    ans1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: s > t and (S + X_t - X_s) % M == 0  => X_s = (X_t + S) % M
    # This is Sum_{j=1 to N} (count of i < j such that X_i = (X_j + S) % M)
    # We can use reduce to maintain the state (current_counts, total_sum)
    
    def update_state(state, x):
        curr_counts, total = state
        target = (x + S_mod) % M
        # We want i < j such that X_i = (X_j + S) % M
        # Here x is X_j, so we look for X_i = (x + S_mod) % M
        # Wait, the condition was s > t. If j is s and i is t:
        # X_j = (X_i + S) % M  => X_i = (X_j - S) % M
        target = (x - S_mod) % M
        curr_counts[x] = curr_counts.get(x, 0) + 1
        return (curr_counts, total + curr_counts.get(target, 0) - (1 if target == x else 0))
    
    # The above logic is slightly flawed because it adds the current x to the count 
    # before checking the target. Let's fix it:
    def update_state_fixed(state, x):
        curr_counts, total = state
        target = (x - S_mod) % M
        # Count how many i < j have X_i = target
        count_target = curr_counts.get(target, 0)
        curr_counts[x] = curr_counts.get(x, 0) + 1
        return (curr_counts, total + count_target)

    # To avoid the 'state' overhead in reduce, we can use a simpler approach for Part 2:
    # Sum_{i < j} I(X_j - X_i == S % M) is NOT what we want.
    # We want Sum_{s > t} I(S + X_t - X_s == 0 mod M)
    # Let s = j, t = i (i < j). We want (S + X_i - X_j) % M == 0
    # => X_j = (S + X_i) % M.
    
    # Let's use the property:
    # Total pairs (s, t) with s != t is:
    # Sum_{i < j} [I(X_i == X_j) + I(X_j == (X_i + S) % M)]
    
    # The first part is Sum C[v]*(C[v]-1)//2.
    # The second part is Sum_{i < j} I(X_j == (X_i + S) % M).
    # This second part can be calculated by:
    # For each i, we need to count j > i such that X_j = (X_i + S) % M.
    # This is (Total count of (X_i + S)%M) - (count of j <= i such that X_j = (X_i + S)%M).
    
    # Let's use a list comprehension with a helper function to simulate the accumulation.
    # Since