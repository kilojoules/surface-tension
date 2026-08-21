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
    # P_3 = A_1 + A_2
    # ...
    # P_i = sum(A_j for j from 1 to i-1)
    # The distance from s to t (s < t) is (P_t - P_s) mod M == 0
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t) mod M == 0
    
    # Calculate prefix sums modulo M
    # accumulate([0] + A) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums (P_1 to P_N)
    prefixes = list(accumulate([0] + A, lambda x, y: (x + y) % M))[:N]
    
    # Total sum of all A_i modulo M
    total_sum_mod = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    # Using a list as a frequency array since M <= 10^6
    counts = [0] * M
    for p in prefixes:
        counts[p] += 1
        
    # For a fixed s and t (s != t):
    # If s < t: (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # If s > t: (Total - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total % M
    
    # Case 1: s < t. For each remainder r, if there are 'c' occurrences,
    # there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # For a fixed remainder r, any two indices i, j with P_i == P_j == r
    # will form a valid pair if we order them such that s < t.
    # So for each r, we have counts[r] * (counts[r] - 1) // 2 pairs.
    # Wait, the logic above is for s < t. What about s > t?
    # If s > t, we need (P_s - P_t) % M == total_sum_mod.
    # Let r_t = P_t % M and r_s = P_s % M.
    # We need (r_s - r_t) % M == total_sum_mod.
    # This means r_s = (r_t + total_sum_mod) % M.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t:
    # If s < t, condition is P_t ≡ P_s (mod M)
    # If s > t, condition is P_t ≡ P_s - Total (mod M)
    
    # Total pairs = Sum_{r=0 to M-1} (count[r] * count[(r - total_sum_mod) % M])
    # But we must exclude cases where s = t.
    # s = t happens when (P_s - P_s) % M == total_sum_mod % M, 
    # which means total_sum_mod % M == 0.
    # If total_sum_mod % M == 0, then for every s, the pair (s, s) is counted.
    # There are N such pairs.
    
    # The number of pairs (s, t) with s != t is:
    # Sum_{r=0 to M-1} (count[r] * count[(r - total_sum_mod) % M])
    # minus (N if total_sum_mod % M == 0 else 0)
    
    # Using a list comprehension to calculate the sum for all r
    ans = sum(counts[r] * counts[(r - total_sum_mod) % M] for r in range(M))
    
    if total_sum_mod == 0:
        ans -= N
        
    print(ans)

if __name__ == "__main__":
    solve()