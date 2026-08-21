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
    
    # Let P_i be the prefix sum of distances: 
    # P_i = sum(A_j for j from 1 to i-1)
    # The distance from s to t (s < t) is (P_t - P_s) mod M == 0
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t) mod M == 0
    
    # Calculate prefix sums modulo M
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # We use accumulate to get [0, A_1, A_1+A_2, ...]
    # Since we only need the first N prefix sums (P_1 to P_N)
    # P_1 = 0
    # P_2 = A_1 % M
    # ...
    # P_N = (A_1 + ... + A_{N-1}) % M
    
    # We create a list of prefix sums modulo M for indices 1 to N
    # prefix_sums[i] represents the distance from rest area 1 to rest area i+1
    prefix_sums = list(accumulate([0] + A[:N-1], lambda x, y: (x + y) % M))
    
    # Total sum of all A_i modulo M
    total_sum_mod = sum(A) % M
    
    # For a pair (s, t) with s < t:
    # (P_t - P_s) % M == 0  => P_t % M == P_s % M
    
    # For a pair (s, t) with s > t:
    # (Total_Sum - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total_Sum % M
    # => P_s % M == (P_t + Total_Sum) % M
    
    # Count occurrences of each remainder
    # Using a dictionary or a list since M <= 10^6
    counts = {}
    for x in prefix_sums:
        counts[x] = counts.get(x, 0) + 1
        
    # For s < t, we need pairs with the same remainder.
    # If a remainder r appears C times, there are C*(C-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If s < t and P_s == P_t, that's one pair.
    # If s > t and P_s == (P_t + Total_Sum) % M, that's another.
    
    # Let's use a different approach to avoid double counting and handle s > t:
    # We want pairs (s, t) such that dist(s, t) % M == 0.
    # dist(s, t) = (P_t - P_s) % Total_Length if s < t
    # dist(s, t) = (Total_Length - P_s + P_t) % Total_Length if s > t
    
    # Actually, the condition is simply:
    # (P_t - P_s) % M == 0 if s < t
    # (Total_Sum + P_t - P_s) % M == 0 if s > t
    
    # Let's use the property: 
    # For every s, we seek t != s such that:
    # 1. t > s and P_t % M == P_s % M
    # 2. t < s and P_t % M == (P_s - Total_Sum) % M
    
    # Total pairs = Sum_{r=0 to M-1} (count(r) * count((r - Total_Sum) % M))
    # But we must exclude cases where s == t.
    # s == t happens when r == (r - Total_Sum) % M, which means Total_Sum % M == 0.
    # In that case, we subtract N from the total.
    
    # Using a list comprehension to iterate over the unique remainders present
    distinct_rs = counts.keys()
    ans = sum(counts[r] * counts.get((r - total_sum_mod) % M, 0) for r in distinct_rs)
    
    # Subtract cases where s == t
    # s == t is counted if (P_s - P_s) % M == (Total_Sum % M == 0)
    # If total_sum_mod == 0, then for every s, (P_s - P_s) % M == 0 is true.
    if total_sum_mod == 0:
        ans -= N
        
    print(ans)

if __name__ == "__main__":
    solve()