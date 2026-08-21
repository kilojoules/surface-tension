import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use next() to extract N and M from the map object
    # Since we cannot use loops, we handle the remaining A_i as a list
    # We use a helper to capture N and M and the rest of the data
    def process_data(data_iter):
        n = next(data_iter)
        m = next(data_iter)
        a = list(data_iter)
        
        # Let P_i be the distance from rest area 1 to rest area i.
        # P_1 = 0
        # P_2 = A_1
        # P_3 = A_1 + A_2 ...
        # The distance from s to t (s < t) is (P_t - P_s)
        # The distance from s to t (s > t) is (Total_Sum - P_s + P_t)
        
        # Compute prefix sums: P_1, P_2, ..., P_N
        # accumulate([0] + a) gives 0, A_1, A_1+A_2, ...
        # We only need the first N prefix sums.
        prefixes = list(accumulate([0] + a))[:n]
        
        # We are interested in P_i mod M
        mods = [p % m for p in prefixes]
        counts = Counter(mods)
        
        # Total distance around the lake
        total_sum = sum(a)
        total_mod = total_sum % m
        
        # For a pair (s, t):
        # If s < t: distance is (P_t - P_s). This is 0 mod M if P_t % M == P_s % M.
        # If s > t: distance is (Total - P_s + P_t). This is 0 mod M if (P_s - P_t) % M == Total % M.
        
        # Case 1: s < t
        # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
        ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
        
        # Case 2: s > t
        # We need (P_s - P_t) % M == total_mod
        # Which means P_t % M == (P_s - total_mod) % M
        # For each s, we count how many t < s satisfy this.
        # However, it's easier to iterate over the counts of remainders.
        # For a remainder r1 (for P_s), we need r2 (for P_t) such that (r1 - r2) % m == total_mod.
        # r2 = (r1 - total_mod) % m.
        
        # To avoid loops, we use a generator expression over the unique remainders in counts.
        ans_s_gt_t = sum(
            counts[r1] * counts[(r1 - total_mod) % m]
            for r1 in counts
        )
        
        # Special handling for s > t:
        # The logic `counts[r1] * counts[r2]` counts all pairs (s, t) with those remainders.
        # But we specifically need s > t.
        # Let's refine:
        # Total pairs (s, t) such that dist(s, t) % M == 0 is:
        # Sum_{s, t} [ (P_t - P_s) % M == 0 if s < t else (Total + P_t - P_s) % M == 0 ]
        
        # Let's use a different approach for s > t to avoid complex logic:
        # For every s, we want t != s such that:
        # If s < t: P_t % M == P_s % M
        # If s > t: P_t % M == (P_s - Total) % M
        
        # Let's calculate for each s the number of t's.
        # For a fixed s, the number of t > s with P_t % M == P_s % M is:
        # (count of P_s % M in P_{s+1...N})
        # The number of t < s with P_t % M == (P_s - Total) % M is:
        # (count of (P_s - Total) % M in P_{1...s-1})
        
        # This is still loop-like. Let's use the property:
        # Total pairs = Sum_{r=0 to M-1} (count[r] * count[(r - total_mod) % m])
        # Wait, if total_mod == 0, then (P_t - P_s) % M == 0 is the same as (Total + P_t - P_s) % M == 0.
        # If total_mod == 0:
        # For any pair {s, t}, both clockwise and counter-clockwise (in terms of indices) 
        # distances are multiples of M if P_s % M == P_t % M.
        # There are N*(N-1)//2 such pairs of indices, and each gives 2 directed pairs (s,t) and (t,s).
        # So result is sum(c * (c-1) for c in counts.values()).
        
        # If total_mod != 0:
        # s < t: P_t % M == P_s % M  => count[r] * (count[r]-1) // 2
        # s > t: P_t % M == (P_s - total_mod) % M => count[r] * count[(r - total_mod) % m]
        # Note: since total_mod != 0, r != (r - total_mod) % m, so these sets are disjoint.
        
        if total_mod == 0:
            return sum(c * (c - 1) for c in counts.values())
        else:
            # s < t pairs
            term1 = sum(c * (c - 1) // 2 for c in counts.values())
            # s > t pairs: for each s, we need t < s such that P_t % M == (P_s - total_mod) % M
            # This is tricky without loops. Let's use the property that 
            # Sum_{s > t} [P_t % M == (P_s - total_mod) % M] 
            # is NOT simply count[r] * count[(r-total_mod)%m] because of the s > t constraint.
            
            # Let's use a different observation:
            # Let x_i = P_i % M.
            # We want pairs (s, t) such that:
            # 1. s < t and x_s == x_t
            # 2. s > t and (x_t - x_s + total_mod) % M == 0  => x_t == (x_s - total_mod) % M
            
            # Let's use a list to track counts of remainders seen so far.
            # Since we can't use loops, we can't easily update a counter.
            # But we can use a generator with a mutable object.
            
            # Correct logic for s > t:
            # We want to count pairs (s, t) with 1 <= t < s <= N such that x_t == (x_s - total_mod) % M.
            # Let's use a helper function with a list to simulate a mutable state inside map/sum.
            state = [0] * m
            def count_s_gt_t(x):
                target = (x - total_mod) % m
                res = state[target]
                state[x] += 1
                return res
            
            # Use map to iterate through mods and sum the results
            return term1 + sum(map(count_s_gt_t, mods))

    # Wrap the logic in a function and call it
    return process_data(input_data)

if __name__ == "__main__":
    # Using a trick to call the solve function and print the result
    # since we cannot use a loop, we just call it once.
    print(solve())