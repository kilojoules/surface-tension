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
    # This simplifies to:
    # If s < t: P_s % M == P_t % M
    # If s > t: (P_s - Total_Sum) % M == P_t % M
    
    # Calculate prefix sums modulo M
    # P = [0, A1, A1+A2, ..., A1+...+A_{N-1}]
    # We use accumulate to get [0, A1, A1+A2, ...]
    # Since we only need sums up to N-1 for the P_i values:
    prefixes = list(accumulate([0] + A[:-1], lambda x, y: (x + y) % M))
    
    # Total sum of all A_i modulo M
    total_sum_mod = sum(A) % M
    
    # For a fixed s, we want to count t != s such that dist(s, t) % M == 0.
    # Let x = P_s % M.
    # We need t such that:
    # 1. t > s and P_t % M == x
    # 2. t < s and (total_sum_mod - x + P_t) % M == 0  => P_t % M == (x - total_sum_mod) % M
    
    # Let count_map be the frequency of each remainder in prefixes
    # We can use a list since M is at most 10^6
    counts = [0] * M
    for p in prefixes:
        counts[p] += 1
        
    # For each s, the number of valid t is:
    # (count of P_t % M == P_s % M) - 1  <-- The -1 excludes t=s
    # PLUS
    # (count of P_t % M == (P_s - total_sum_mod) % M)
    # BUT, if (P_s % M) == (P_s - total_sum_mod) % M, we have double counted t=s.
    # However, the logic above is for a specific s. 
    # Let's refine:
    # For a fixed s, we seek t != s such that:
    # If t > s: P_t % M == P_s % M
    # If t < s: P_t % M == (P_s - total_sum_mod) % M
    
    # This is tricky because the condition depends on whether t > s.
    # Let's use the property:
    # Total pairs = sum_{s=1}^N (count of t > s where P_t == P_s) 
    #              + sum_{s=1}^N (count of t < s where P_t == (P_s - total_sum_mod))
    
    # Let f(v) be the number of times remainder v appears in prefixes.
    # The number of pairs (s, t) with s < t and P_s == P_t is f(v)*(f(v)-1)//2 summed over v.
    # The number of pairs (s, t) with s > t and P_t == (P_s - total_sum_mod) is:
    # For each s, we need t < s. This is harder.
    
    # Let's reconsider:
    # A pair (s, t) is valid if:
    # 1. s < t and (P_t - P_s) % M == 0
    # 2. s > t and (Total - P_s + P_t) % M == 0
    
    # Let X = P_s % M and Y = P_t % M.
    # Condition 1: X == Y and s < t
    # Condition 2: Y == (X - Total) % M and s > t
    
    # Let's use the fact that for any two distinct indices i, j (i < j):
    # They form a valid pair (s=i, t=j) if P_i == P_j
    # They form a valid pair (s=j, t=i) if P_i == (P_j - Total) % M
    
    # Total = sum_{i < j} [P_i == P_j] + sum_{i < j} [P_i == (P_j - Total) % M]
    
    # sum_{i < j} [P_i == P_j] is sum (f(v) * (f(v)-1) // 2)
    
    # For the second term: sum_{j=1}^N (count of i < j such that P_i == (P_j - Total) % M)
    # We can calculate this by iterating j from 0 to N-1 and maintaining a running count of P_i.
    
    # First term:
    ans = sum(c * (c - 1) // 2 for c in counts if c > 1)
    
    # Second term:
    # We need sum_{j=0}^{N-1} (count of i < j where P_i == (P_j - total_sum_mod) % M)
    # We use a list comprehension with a custom function or a loop.
    # Since we cannot use loops, we can use a map/reduce approach to track state.
    
    def update_state(state, p_j):
        current_counts, total_valid = state
        target = (p_j - total_sum_mod) % M
        # Count how many P_i (i < j) match the target
        valid_for_j = current_counts[target]
        # Update count for P_j
        # Note: lists are mutable, so we update in place
        current_counts[p_j] += 1
        return (current_counts, total_valid + valid_for_j)

    # Initialize state: ([0]*M, 0)
    # We use reduce to simulate the loop over j
    final_state = reduce(update_state, prefixes, ([0] * M, 0))
    
    print(ans + final_state[1])

if __name__ == "__main__":
    solve()