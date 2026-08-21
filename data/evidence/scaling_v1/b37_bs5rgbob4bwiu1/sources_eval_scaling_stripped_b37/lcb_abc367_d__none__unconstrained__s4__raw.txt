```python
import sys
from itertools import accumulate
from functools import reduce

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of distances: P_i = sum(A_1 ... A_{i-1})
    # The distance from s to t (s < t) is (P_t - P_s) mod M == 0
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t) mod M == 0
    # Which is (P_t - P_s) % M == 0 if we consider the total sum modulo M.
    # Wait, the condition is: (P_t - P_s) % M == 0 for s < t
    # And (Total_Sum + P_t - P_s) % M == 0 for s > t.
    
    # Let X_i = P_i % M. 
    # For s < t: (X_t - X_s) % M == 0  => X_t == X_s
    # For s > t: (Total_Sum + X_t - X_s) % M == 0 => (X_s - X_t) % M == Total_Sum % M
    
    # Prefix sums modulo M
    # P_1 = 0, P_2 = A_1, P_3 = A_1 + A_2 ...
    # We use accumulate to get [0, A_1, A_1+A_2, ...]
    # We only need N values (for rest areas 1 to N)
    prefixes = list(accumulate(A, initial=0))[:N]
    X = [x % M for x in prefixes]
    
    # Count occurrences of each remainder
    # Using a dictionary or a list since M <= 10^6
    counts = {}
    for x in X:
        counts[x] = counts.get(x, 0) + 1
    
    # Total sum of A modulo M
    S = sum(A) % M
    
    # For a fixed s and t:
    # If s < t, we need X_t == X_s
    # If s > t, we need X_s - X_t == S (mod M)
    
    # Let's use the property: 
    # Total pairs = Sum_{i, j} [ (X_j - X_i) % M == 0 if i < j else (X_j - X_i + S) % M == 0 ]
    # This is tricky because of the i < j condition.
    # Let's rewrite:
    # For every pair {i, j} with i < j:
    # Check if (X_j - X_i) % M == 0  (s=i, t=j)
    # Check if (X_i - X_j + S) % M == 0 (s=j, t=i)
    
    # Let C_v be the number of times remainder v appears in X.
    # The number of pairs (i, j) with i < j such that X_i == X_j is:
    # Sum_{v} (C_v * (C_v - 1) // 2)
    
    # The number of pairs (i, j) with i < j such that (X_i - X_j + S) % M == 0:
    # This is X_i - X_j == S (mod M)  => X_j == (X_i - S) (mod M)
    # For a fixed i, we need the number of j > i such that X_j == (X_i - S) % M.
    # This is hard to do with just global counts.
    
    # Actually, let's use the property:
    # Total = Sum_{i < j} [X_i == X_j] + Sum_{i < j} [X_j == (X_i - S) % M]
    # Wait, the second term is: for each i, count j > i where X_j == (X_i - S) % M.
    # This is equivalent to: for each value v, count pairs (i, j) with i < j 
    # such that X_i = v and X_j = (v - S) % M.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t:
    # Distance is (P_t - P_s) mod Total_Sum.
    # We want (P_t - P_s) % M == 0 if s < t
    # We want (Total_Sum + P_t - P_s) % M == 0 if s > t
    
    # Let X_i = P_i % M.
    # s < t: X_t - X_s \equiv 0 \pmod M  => X_t \equiv X_s \pmod M
    # s > t: X_t - X_s \equiv -S \pmod M => X_t \equiv X_s - S \pmod M
    
    # Let's count for all i, j:
    # If X_i == X_j, then (i, j) is a pair if i < j.
    # If X_j == (X_i - S) % M, then (i, j) is a pair if i > j.
    
    # Let's use the counts:
    # For a fixed value v, there are C_v instances.
    # Pairs (i, j) with i < j and X_i = X_j = v: C_v * (C_v - 1) // 2
    # Pairs (i, j) with i > j and X_i = v, X_j = (v - S) % M:
    # This is the number of pairs (j, i) with j < i and X_j = (v - S) % M and X_i = v.
    
    # Let v2 = (v - S) % M.
    # We want number of pairs (j, i) such that j < i, X_j = v2, X_i = v.
    # This depends on the relative positions.
    
    # Let's use the fact that:
    # Total = Sum_{v} (Pairs i < j with X_i=X_j=v) + Sum_{v} (Pairs j < i with X_j=(v-S)%M, X_i=v)
    # The second term is: Sum_{v} (Pairs j < i with X_j=v2, X_i=v)
    # Note that Sum_{j < i} [X_j=v2 and X_i=v] + Sum_{j > i} [X_j=v2 and X_i=v] = C_{v2} * C_v
    # (Assuming v != v2).
    
    # Let's simplify:
    # We want Sum_{i < j} [X_i == X_j] + Sum_{j < i} [X_j == (X_i - S) % M]
    # Let v2 = (X_i - S) % M.
    # The second term is Sum_{i} (count of j < i such that X_j == (X_i - S) % M).
    
    # Let's process the array X and maintain counts of values seen so far.
    # For each X_i:
    # 1. Add (count of X_i seen so far) to total.
    # 2. Add (count of (X_i - S) % M seen so far) to total.
    # Wait, the second one is: j < i and X_j == (X_i - S) % M. This is exactly what we need!
    # But we must be careful: if S % M == 0, then (X_i - S) % M == X_i.
    # The condition s != t is given.
    # If S % M == 0:
    # s < t: X_s == X_t
    # s > t: X_s == X_t
    # So for every pair {i, j} with X_i == X_j, both (i, j) and (j, i) work.
    # Total = 2 * Sum (C_v * (C_v - 1) // 2) = Sum (C_v * (C_v - 1))
    
    # If S % M != 0:
    # s < t: X_s == X_t
    # s > t: X_j == (X_i - S) % M where j < i
    # Total = Sum_{i < j} [X_i == X_j] + Sum_{j < i} [X_j == (X_i - S) % M]
    # The first term is Sum (C_v * (C_v - 1) // 2)
    # The second term: For each i, count j < i such that X_j == (X_i - S) % M.
    # This is Sum_{i} (count of (X_i - S) % M in X[0...i-1])
    
    # Let's implement this.
    
    # To avoid loops, we can use a list comprehension and a dictionary/list for counts.
    # Since we can't use loops, we can use a custom function with reduce.
    
    def update_state(state, x):
        current_count, total_found = state
        # We need to track counts of all remainders. 
        # Since we can't use a dictionary in a reduce without mutation, 
        # we use a list/array for counts and a dictionary to store the state.
        # Actually, we can use a dictionary and return a new one, but that's O(N) per step.
        # We must use a mutable object (like a list) inside the reduce to keep track of counts.
        return state

    # Wait, the constraints on loops are strict. But I can use a list for counts 
    # and a reduce function that updates the list and returns the running total.
    
    # Correct logic for S % M != 0:
    # Total = Sum_{v} (C_v * (C_v - 1) // 2) + Sum_{i=0}^{N-1} (count of (X_i - S) % M in X[0...i-1])
    
    # Let's use a list for counts: count_list = [0] * M
    # We can use a function that updates the list and returns the count of (X_i - S) % M.
    
    # Since I cannot use a for loop, I will use map/reduce.
    # To maintain state (the count list), I'll use a list of size M and update it.
    
    # However, the most efficient way to calculate Sum_{j < i} [X_j == (X_i - S) % M]
    # without a loop is to use the fact that:
    # Sum_{j < i} [X_j == v2 and X_i == v] = (Total pairs with X_j=v2, X_i=v) 
    #                                      - (Pairs with j > i and X_j=v2, X_i=v)
    # This doesn't help.
    
    # Let's use the property:
    # Sum_{j < i} [X_j == v2 and X_i == v] 
    # If we have the indices of all occurrences of v and v2:
    # For a fixed i where X_i = v, we need count of j < i where X_j = v2.
    # This is a classic problem that can be solved by sorting or using the fact that
    # we only need the total sum.
    
    # Actually, the simplest way to calculate Sum_{j < i} [X_j == v2 and X_i == v]
    # is to use the total counts C_v and C_{v2} and the average distribution? No.
    # But wait! The problem is symmetric.
    # Sum_{j < i} [X_j == v2 and X_i == v] + Sum_{j > i} [X_j == v2 and X_i == v] = C_{v2} * C_v
    # And Sum_{j > i} [X_j == v2 and X_i == v] is exactly the same as 
    # Sum_{i < j} [X_i == v and X_j == v2].
    
    # Let's use the fact that we can use a list comprehension to create a list