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
    
    # Let P_i be the prefix sum of distances from rest area 1.
    # P_1 = 0
    # P_2 = A_1
    # P_i = A_1 + ... + A_{i-1}
    # The distance from s to t (s < t) is (P_t - P_s) mod M == 0
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t) mod M == 0
    
    # Calculate prefix sums modulo M
    # accumulate([0]) + A gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums (for indices 1 to N)
    prefixes = list(accumulate([0] + A))[:-1]
    mods = [p % M for p in prefixes]
    
    # Count occurrences of each remainder modulo M
    # Using a list comprehension to count frequencies manually without loops
    # Since we cannot use loops, we use a dictionary or a frequency array.
    # We can use a map/reduce approach to count frequencies.
    from collections import Counter
    counts = Counter(mods)
    
    # For a fixed s and t (s < t), we need P_t % M == P_s % M.
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t) where s != t.
    # If s < t, we need (P_t - P_s) % M == 0.
    # If s > t, we need (Total_Sum + P_t - P_s) % M == 0.
    
    # Let S = Total_Sum % M
    total_sum_mod = sum(A) % M
    
    # For s < t: P_t % M == P_s % M
    # For s > t: P_s % M == (P_t + S) % M
    
    # We can use a generator expression with sum() to avoid explicit loops.
    # For every remainder r, the number of pairs (s, t) with s < t is:
    # count(r) * (count(r) - 1) // 2
    # But we need to consider both s < t and s > t.
    
    # Let's use the property: 
    # Total pairs = sum_{r} (count(r) * count((r + S) % M))
    # Wait, that's not quite right because s != t.
    # If S == 0, then (r + S) % M == r, so we get count(r)^2. 
    # But we must exclude s == t, so count(r)^2 - count(r).
    # If S != 0, then r != (r + S) % M, so we just get count(r) * count((r + S) % M).
    
    # Let's refine:
    # We want pairs (s, t) such that dist(s, t) % M == 0.
    # dist(s, t) = (P_t - P_s) % Total_Length if s < t
    # dist(s, t) = (Total_Length - P_s + P_t) % Total_Length if s > t
    
    # Actually, the condition is simply:
    # (P_t - P_s) % M == 0 if s < t
    # (P_t - P_s + Total_Sum) % M == 0 if s > t
    
    # Let f(r) be the number of i such that P_i % M == r.
    # For a fixed r, there are f(r) indices.
    # Pairs (s, t) with s < t and P_s % M == P_t % M:
    # This is sum_{r} f(r)*(f(r)-1)//2
    # Pairs (s, t) with s > t and (P_t - P_s + S) % M == 0:
    # This is P_s % M == (P_t + S) % M.
    # For a fixed t, we need s > t such that P_s % M == (P_t + S) % M.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t:
    # If s < t, condition is P_s % M == P_t % M
    # If s > t, condition is P_s % M == (P_t + S) % M
    
    # Total = sum_{s < t} [P_s % M == P_t % M] + sum_{s > t} [P_s % M == (P_t + S) % M]
    # The first term is sum_{r} f(r)*(f(r)-1)//2
    # The second term: for each t, we need s > t such that P_s % M == (P_t + S) % M.
    # This is tricky because of the s > t constraint.
    
    # Let's use the fact that:
    # sum_{s < t} [P_s % M == P_t % M] + sum_{s > t} [P_s % M == P_t % M] = sum f(r)*(f(r)-1)
    # The second term of our goal is sum_{s > t} [P_s % M == (P_t + S) % M].
    
    # Let's use a different logic:
    # For every pair (s, t) with s != t, the clockwise distance is:
    # (P_t - P_s) mod (Total_Sum)
    # We want (P_t - P_s) % M == 0 (if s < t) OR (Total_Sum + P_t - P_s) % M == 0 (if s > t).
    # This is equivalent to:
    # If s < t: P_t % M == P_s % M
    # If s > t: P_s % M == (P_t + Total_Sum) % M
    
    # Let's use a list comprehension to calculate the answer:
    # We iterate over all possible remainders r from 0 to M-1.
    # For a fixed r, let c1 = count(r) and c2 = count((r + Total_Sum) % M).
    # The number of pairs (s, t) with s < t and P_s % M == P_t % M is sum(c*(c-1)//2).
    # The number of pairs (s, t) with s > t and P_s % M == (P_t + S) % M is...
    # Actually, the simplest way is:
    # For every pair (s, t) with s < t:
    # Check if P_s % M == P_t % M  (this is s -> t)
    # Check if P_s % M == (P_t + S) % M (this is t -> s)
    
    # Total = sum_{r=0}^{M-1} [ f(r)*f(r)-1 // 2  +  (count pairs s < t where P_s % M == (P_t + S) % M) ]
    # This is still confusing. Let's use the property:
    # For any two distinct indices i, j (i < j):
    # Pair (i, j) is valid if P_i % M == P_j % M
    # Pair (j, i) is valid if P_j % M == (P_i + S) % M
    
    # Total = sum_{i < j} ([P_i % M == P_j % M] + [P_j % M == (P_i + S) % M])
    # Total = sum_{r=0}^{M-1} (f(r)*(f(r)-1)//2) + sum_{i < j} [P_j % M == (P_i + S) % M]
    
    # To calculate sum_{i < j} [P_j % M == (P_i + S) % M] without loops:
    # We can use the fact that:
    # sum_{i < j} [P_j % M == (P_i + S) % M] + sum_{i > j} [P_j % M == (P_i + S) % M] 
    # = sum_{i != j} [P_j % M == (P_i + S) % M]
    # = sum_{r=0}^{M-1} f(r) * f((r + S) % M) - (if S == 0 then sum f(r) else 0)
    
    # But we only want i < j. This is hard without loops.
    # Wait! The problem can be solved by:
    # For every pair (s, t) with s != t, the distance is a multiple of M iff:
    # (P_t - P_s) % M == 0 if s < t
    # (P_t - P_s + S) % M == 0 if s > t
    
    # Let's use a different approach.
    # For every i, we want to count j != i such that dist(i, j) % M == 0.
    # If j > i, we need P_j % M == P_i % M.
    # If j < i, we need P_j % M == (P_i + S) % M.
    
    # Let's use the fact that we can process the array and keep track of counts.
    # Since we can't use loops, we can use a custom function with reduce.
    # state = (current_counts_dict, total_valid_pairs)
    
    def update(state, p_mod):
        counts, total = state
        # For the current j = i, we count s < j such that P_s % M == P_j % M
        # And we count s > j... no, that's not how it works.
        # For the current j, the number of s < j such that dist(s, j) % M == 0 is counts[p_mod]
        # The number of s < j such that dist(j, s) % M == 0 is counts[(p_mod + S) % M]
        # Wait, if s < j, dist(j, s) = (S + P_s - P_j) % M.
        # So we need P_s % M == (P_j - S) % M.
        
        # Correct logic for index j:
        # Pairs (s, j) with s < j: valid if P_s % M == P_j % M
        # Pairs (j, s) with s < j: valid if P_s % M == (P_j - S) % M
        
        # Update counts dictionary
        new_counts = counts.copy()
        new_counts[p_mod] = new_counts.get(p_mod, 0) + 1
        
        # Current j is the second element in the pair.
        # s < j and dist(s, j) % M == 0  => P_s % M == P_j % M
        # s < j and dist(j, s) % M == 0  => P_s % M == (P_j - S) % M
        
        # We use .get(key, 0) to avoid KeyErrors
        gain = counts.get(p_mod, 0) + counts.get((p_mod - S) % M, 0)
        
        # Special case: if S % M == 0, then (P_j % M) == (P_j - S) % M.
        # The two conditions are the same. But the problem says s != t.
        # If S % M == 0, then dist(s, j) % M == 0 iff dist(j, s) % M == 0.
        # The logic above counts both.
        
        return (new_counts, total + gain)

    # Initial state: (empty dict, 0)
    # We use reduce to simulate the loop over mods
    # S is total_sum_mod
    S = total_sum_mod
    final_state = reduce(
        lambda state, p_mod: (
            {**state[0], p_mod: state[0].get(p_mod, 0) + 1},
