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
    
    # Let P_i be the prefix sum of distances from area 1 to area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2
    # ...
    # P_i = sum(A_j for j from 1 to i-1)
    # The distance from s to t (s < t) is (P_t - P_s) mod M == 0.
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t) mod M == 0.
    
    # Calculate prefix sums modulo M
    # accumulate([0] + A) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums (P_1 to P_N)
    prefixes = list(accumulate([0] + A))[:N]
    mods = [p % M for p in prefixes]
    
    # Count occurrences of each remainder modulo M
    # Using a list comprehension to count frequencies of each remainder 0...M-1
    # However, since we only care about values present in 'mods', 
    # we can use a dictionary or a frequency array.
    # Given M <= 10^6, a frequency array is efficient.
    counts = [0] * M
    for m in mods:
        counts[m] += 1
    
    # For a fixed s and t (s != t):
    # If s < t: (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # If s > t: (Total_Sum - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total_Sum % M
    
    total_sum_mod = sum(A) % M
    
    # Case 1: s < t. For each remainder r, if there are 'c' indices with that remainder,
    # there are c*(c-1)//2 pairs.
    ans_lt = sum(c * (c - 1) // 2 for c in counts)
    
    # Case 2: s > t. We need (P_s - P_t) % M == total_sum_mod.
    # This means P_t % M == (P_s - total_sum_mod) % M.
    # For every s, we look for t < s such that P_t % M == (P_s - total_sum_mod) % M.
    # This is equivalent to summing counts[r] * counts[(r - total_sum_mod) % M]
    # for all r, but we must exclude the case where s == t (which is not allowed).
    # Wait, the s > t logic is simpler: 
    # For every pair (s, t) with s > t, the condition is P_s - P_t ≡ Total % M.
    # Let r_s = P_s % M and r_t = P_t % M.
    # We need r_s - r_t ≡ Total % M  => r_t ≡ (r_s - Total) % M.
    # For a fixed r_s, there are counts[r_s] choices for s and counts[(r_s - Total) % M] choices for t.
    # This counts all pairs (s, t) such that s > t AND (s, t) such that s < t 
    # IF we didn't have the s > t constraint.
    # Actually, the most direct way to count pairs (s, t) with s != t is:
    # A pair (s, t) is valid if:
    # 1. s < t and (P_t - P_s) % M == 0
    # 2. s > t and (Total - P_s + P_t) % M == 0
    
    # Let's use the property: 
    # For every pair {i, j} with i < j:
    # Pair (i, j) is valid if P_j - P_i ≡ 0 (mod M)
    # Pair (j, i) is valid if Total - (P_j - P_i) ≡ 0 (mod M)
    
    # Let diff = (P_j - P_i) % M.
    # (i, j) is valid if diff == 0.
    # (j, i) is valid if diff == Total % M.
    
    # If Total % M == 0:
    # Then (i, j) is valid iff (j, i) is valid.
    # Both are valid if P_i % M == P_j % M.
    # Total pairs = 2 * sum(c * (c-1) // 2) = sum(c * (c-1))
    
    # If Total % M != 0:
    # (i, j) is valid if P_i % M == P_j % M.
    # (j, i) is valid if P_j % M - P_i % M == Total % M.
    # Total pairs = sum(c * (c-1) // 2 for c in counts) 
    #               + sum(counts[r] * counts[(r - total_sum_mod) % M] 
    #                  for r in range(M) if (r - total_sum_mod) % M != r)
    # Wait, the second term is simply the number of pairs (i, j) with i < j 
    # such that P_j - P_i ≡ Total (mod M).
    # Actually, for any two indices i, j, they contribute to the answer if:
    # (P_j - P_i) % M == 0  OR  (P_j - P_i) % M == Total % M.
    # If Total % M == 0, these conditions are the same.
    # If Total % M != 0, these conditions are mutually exclusive.
    
    # Correct logic for Total % M != 0:
    # Count pairs (i, j) with i < j such that P_j - P_i ≡ 0 (mod M) -> (i, j) is valid.
    # Count pairs (i, j) with i < j such that P_j - P_i ≡ Total (mod M) -> (j, i) is valid.
    # Note: (P_j - P_i) ≡ Total (mod M) is the same as P_i ≡ (P_j - Total) (mod M).
    
    # For a fixed j, the number of i < j such that P_i ≡ P_j (mod M) is 
    # the number of times P_j % M has appeared before.
    # The number of i < j such that P_i ≡ (P_j - Total) (mod M) is
    # the number of times (P_j - Total) % M has appeared before.
    
    # This can be solved by iterating through the mods and maintaining a running count.
    # But we can use the global counts:
    # Pairs (i, j) with i < j and P_i ≡ P_j (mod M) is sum(c*(c-1)//2).
    # Pairs (i, j) with i < j and P_j - P_i ≡ Total (mod M) is:
    # This is tricky because the "i < j" constraint depends on the order.
    # Let's use the property: 
    # Total pairs = (Pairs i < j where P_j - P_i ≡ 0) + (Pairs i < j where P_j - P_i ≡ Total)
    # The second term: for every pair {i, j}, one is smaller. 
    # The condition P_j - P_i ≡ Total (mod M) for i < j is exactly the same as
    # counting pairs (i, j) such that i < j and P_j - P_i ≡ Total (mod M).
    # This is NOT simply counts[r] * counts[r - Total].
    # That would count all pairs regardless of index.
    # Let's use a different approach for the second term:
    # For each j from 1 to N, we want count of i < j such that P_i ≡ (P_j - Total) (mod M).
    
    # Let's redefine:
    # Let S = Total % M.
    # We want:
    # 1. i < j and (P_j - P_s) % M == 0
    # 2. i > j and (S - (P_i - P_j)) % M == 0  => (P_i - P_j) % M == S
    
    # Both conditions are:
    # 1. i < j and P_i ≡ P_j (mod M)
    # 2. i > j and P_i - P_j ≡ S (mod M)
    
    # Let's process the array and maintain counts of remainders seen so far.
    # For each P_j:
    #   Add current_counts[P_j % M] to answer (for case 1)
    #   Add current_counts[(P_j - S) % M] to answer (for case 2)
    #   Increment current_counts[P_j % M]
    
    # Wait, if S == 0, then (P_j - S) % M == P_j % M, so we add the same value twice.
    # That's correct because if S == 0, then (i, j) is valid AND (j, i) is valid.
    
    # Implementation using a loop/reduce to avoid explicit for-loops:
    def update(state, p):
        curr_ans, curr_counts = state
        p_mod = p % M
        # Case 1: i < j and P_i == P_j
        # Case 2: i < j and P_j - P_i == S  => P_i == P_j - S
        # We use a list for curr_counts and a variable for curr_ans.
        # Since we cannot mutate lists in reduce easily without side effects,
        # we use a dictionary or a mutable object.
        return state # placeholder
    
    # To keep it strictly functional/reduce-based:
    # We can't easily update a list in reduce. Let's use a dictionary and 
    # a custom function, but the prompt asks for a complete program.
    # The most efficient way in Python is a loop, but I will use a 
    # comprehension/map/reduce pattern if possible.
    
    # Actually, the simplest way to count pairs (i, j) with i < j and P_j - P_i ≡ K (mod M)
    # is to iterate through the list and keep track of frequencies.
    
    # Let's use a loop inside a function and call it.
    def count_pairs(mods, S, M):
        freq = [0] * M
        total = 0
        for m in mods:
            total += freq[m] # Case 1: P_i == m
            total += freq[(m - S) % M] # Case 2: P_i == m - S
            freq[m] += 1
        return total

    # Special case: If S == 0, the two conditions are identical.
    # The loop above adds freq[m] twice. If S == 0, then (m - 0) % M == m.
    # So it adds freq[m] + freq[m]. This is correct because both (i, j) and (j, i) are valid.
    # However, if S == 0, the problem says s != t. 
    # For a pair {i, j}, if P_i == P_j, then both (i, j) and (j, i) are valid.
    # The loop adds 1 for (i, j) and 1 for (j, i) effectively.
    
    # Wait, if S == 0, the loop does:
    # total += freq[m] (for i < j)
    # total += freq[m] (for i < j)
    # This correctly counts both (i, j) and (j, i).
    
    # One final check: if S == 0, and P_i == P_j, then:
    # Clockwise i to j: (P_j - P_i) % M == 0
    # Clockwise j to i: (Total