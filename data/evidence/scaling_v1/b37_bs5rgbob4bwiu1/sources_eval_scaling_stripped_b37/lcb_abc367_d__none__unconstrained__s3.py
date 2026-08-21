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
    
    # Calculate prefix sums modulo M
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # We use accumulate to get [A_1, A_1+A_2, ...] then prepend 0
    prefixes = list(accumulate(A, lambda x, y: (x + y) % M))
    # To get P_0 = 0, we can use a list comprehension or insert
    # But accumulate gives us P_1 to P_N. We need P_0 to P_{N-1}.
    # P_0 = 0
    # P_1 = A_1 % M
    # ...
    # P_{N-1} = (A_1 + ... + A_{N-1}) % M
    
    # Correct prefix sums:
    # s is the starting rest area (1 to N)
    # t is the ending rest area (1 to N)
    # Distance s -> t is (Sum_{i=s}^{t-1} A_i) if s < t
    # Distance s -> t is (Sum_{i=s}^{N} A_i + Sum_{i=1}^{t-1} A_i) if s > t
    
    # Let S_i = (sum_{j=1}^{i-1} A_j) mod M. 
    # S_1 = 0
    # S_2 = A_1 % M
    # ...
    # S_N = (A_1 + ... + A_{N-1}) % M
    # Total = (A_1 + ... + A_N) % M
    
    # For s < t: (S_t - S_s) % M == 0  => S_t % M == S_s % M
    # For s > t: (Total - S_s + S_t) % M == 0 => (S_s - S_t) % M == Total % M
    
    # Generate S_1 ... S_N
    # A is A_1 ... A_N. 
    # S_1 = 0. S_i = S_{i-1} + A_{i-1}.
    # We only need S_1 ... S_N.
    # The sequence of A_i used for S is A[0]...A[N-2].
    
    S = [0] + list(map(lambda x: x % M, accumulate(A[:-1])))
    
    # Count occurrences of each remainder in S
    # Using a dictionary to simulate a frequency array
    counts = {}
    for val in S:
        counts[val] = counts.get(val, 0) + 1
        
    total_sum_mod = sum(A) % M
    
    # For a fixed remainder r, there are counts[r] indices.
    # Pairs (s, t) with s < t and S_s == S_t:
    # For each r, we have counts[r] * (counts[r] - 1) // 2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If s < t, we need S_s == S_t.
    # If s > t, we need (S_s - S_t) % M == total_sum_mod.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t:
    # If s < t, condition is S_t - S_s \equiv 0 \pmod M
    # If s > t, condition is S_t - S_s \equiv -Total \pmod M
    
    # This is equivalent to:
    # Count pairs (s, t) such that s < t and S_s == S_t
    # PLUS
    # Count pairs (s, t) such that s > t and S_s - S_t \equiv Total \pmod M
    
    # Let C(r) be the number of times remainder r appears in S.
    # Pairs (s, t) with s < t and S_s == S_t:
    # For each r: C(r) * (C(r) - 1) // 2
    # Wait, the above is only for s < t. But we can have s > t too.
    # Let's refine:
    # We want pairs (s, t) with s != t such that:
    # 1. s < t and S_t \equiv S_s \pmod M
    # 2. s > t and S_t \equiv S_s - Total \pmod M
    
    # For a fixed pair of indices {i, j} with i < j:
    # They contribute to the answer if:
    # (s=i, t=j) and S_j == S_i
    # OR
    # (s=j, t=i) and S_i == (S_j - Total) % M
    
    # Total count = \sum_{i < j} [S_i == S_j] + \sum_{i < j} [S_i == (S_j - Total) % M]
    # This is equivalent to:
    # \sum_{r} (C(r) * (C(r) - 1) // 2)  <-- Case s < t
    # + \sum_{r} (C(r) * C((r + Total) % M)) <-- Case s > t, but we must handle r == (r + Total) % M
    # Wait, the second sum is over all i, j such that i < j and S_i == (S_j - Total) % M.
    # This is not a simple product of counts because of the i < j constraint.
    
    # Let's reconsider:
    # We want pairs (s, t) such that dist(s, t) % M == 0.
    # dist(s, t) = (S_t - S_s) % M if s < t
    # dist(s, t) = (Total - S_s + S_t) % M if s > t
    
    # Let's use the property:
    # (s, t) is valid if:
    # s < t and S_s \equiv S_t \pmod M
    # s > t and S_s \equiv S_t + Total \pmod M
    
    # Let's iterate through all possible remainders r1 and r2:
    # If r1 == r2, we have C(r1) * (C(r1) - 1) // 2 pairs where s < t.
    # If (r1 - r2) % M == Total % M, we have pairs where s > t.
    # This is tricky because the s > t condition depends on the specific indices.
    
    # Actually, the simplest way:
    # For every pair of indices i, j (i < j):
    # Pair (i, j) is valid if S_i == S_j
    # Pair (j, i) is valid if (S_j - S_i) % M == Total % M
    
    # Total = \sum_{i < j} [S_i == S_j] + \sum_{i < j} [S_j - S_i \equiv Total \pmod M]
    # The first term is \sum C(r)(C(r)-1)/2
    # The second term: for a fixed j, we need i < j such that S_i == (S_j - Total) % M.
    # This is exactly what we get if we iterate j from 1 to N and maintain counts of S_i seen so far.
    
    # Let's use a different logic:
    # For every pair of distinct indices i, j:
    # If i < j: valid if S_i == S_j
    # If i > j: valid if S_i - S_j \equiv Total \pmod M
    
    # Let's use the counts C(r).
    # The number of pairs (i, j) with i < j and S_i == S_j is \sum C(r)(C(r)-1)/2.
    # The number of pairs (i, j) with i > j and S_i - S_j \equiv Total \pmod M is:
    # For a fixed i, we need j < i such that S_j \equiv S_i - Total \pmod M.
    # This is \sum_{i=1}^N (count of S_j == (S_i - Total) % M for j < i).
    
    # Let's implement this using a loop and a dictionary.
    
    res = 0
    current_counts = {}
    for val in S:
        # Case s < t: current val is S_t, we need S_s == val
        res += current_counts.get(val, 0)
        # Case s > t: current val is S_s, we need S_t == (val - total_sum_mod) % M
        # Wait, the loop processes indices in increasing order.
        # When we are at index i, 'current_counts' contains S_j for j < i.
        # So for the current S_i:
        # 1. It can be S_t (with s < i), so we need S_s == S_i. 
        #    The number of such s is current_counts[S_i].
        # 2. It can be S_s (with t < i), so we need S_t == (S_i - total_sum_mod) % M.
        #    The number of such t is current_counts[(S_i - total_sum_mod) % M].
        
        # However, if S_i == (S_i - total_sum_mod) % M, then both conditions are the same.
        # But the problem says s != t. If s < t and s > t are both true, that's impossible.
        # The two cases are:
        # 1. s = j, t = i (where j < i)
        # 2. s = i, t = j (where j < i)
        # These are distinct pairs (s, t).
        
        # So for each i, we add:
        # count of j < i such that S_j == S_i
        # count of j < i such that S_j == (S_i - total_sum_mod) % M
        
        # One edge case: if S_i == (S_i - total_sum_mod) % M, then Total % M == 0.
        # In that case, both conditions are S_j == S_i.
        # But we are counting pairs (s, t). (j, i) and (i, j) are different.
        # So we still add both.
        
        # Let's use a generator/map to avoid the explicit loop for speed, 
        # but we need the state of current_counts. 
        # A custom reduce function can handle this.
        pass

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    S = [0] + list(map(lambda x: x % M, accumulate(A[:-1])))
    total_sum_mod = sum(A) % M
    
    # We need sum_{i=1}^N (count(S_j == S_i for j < i) + count(S_j == (S_i - total_sum_mod) % M for j < i))
    # This can be solved by tracking counts in a dictionary and using reduce.
    
    def update(state, s_i):
        counts, total = state
        # count of j < i such that S_j == S_i
        c1 = counts.get(s_i, 0)
        # count of j < i such that S_j == (S_i - total_sum_mod) % M
        c2 = counts.get((s_i - total_sum