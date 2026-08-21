import sys
import math
from itertools import permutations, product
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute printing time for each segment
    # print_times[i] is the time taken to print segment i
    print_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments
    # For each segment, we can start at either endpoint
    # There are N! permutations and 2^N direction combinations
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    
    # For a fixed permutation, we use dynamic programming to find the min time
    # dp[i][dir] = min time to finish first i segments, ending at endpoint 'dir' of segment i
    # dir 0: ended at (C_i, D_i), dir 1: ended at (A_i, B_i)
    
    def get_min_time_for_perm(perm):
        # Initial state: distance from (0,0) to the start of the first segment
        # first_seg = segments[perm[0]]
        # Option 0: (0,0) -> A -> C. Time: dist(0, A)/S + print_times[perm[0]]
        # Option 1: (0,0) -> C -> A. Time: dist(0, C)/S + print_times[perm[0]]
        
        s0 = segments[perm[0]]
        t0 = print_times[perm[0]]
        
        # dp state: (time_ending_at_C, time_ending_at_A)
        initial_dp = (
            dist((0, 0), s0[0]) / S + t0,
            dist((0, 0), s0[1]) / S + t0
        )
        
        # Transition for the rest of the permutation
        def transition(dp, idx):
            s_prev = segments[perm[reduce(lambda x, _: x+1, range(0), 0)]] # This is tricky without loops
            # Instead of reduce for index, we can pass the current index in a custom way
            # But since we are inside a function, we can use a helper
            pass

        # To avoid loops and maintain state, we use a fold-like structure with a generator
        # We'll redefine the DP transition using a list comprehension and a helper
        return 0

    # Since the constraint forbids loops, I will use a recursive-like structure 
    # via reduce to simulate the DP across the permutation.
    
    def calculate_path_time(perm):
        # dp stores (min_time_ending_at_endpoint_0, min_time_ending_at_endpoint_1)
        # endpoint 0 is segments[i][1] (C, D), endpoint 1 is segments[i][0] (A, B)
        
        s0 = segments[perm[0]]
        t0 = print_times[perm[0]]
        start_dp = (dist((0, 0), s0[0]) / S + t0, dist((0, 0), s0[1]) / S + t0)
        
        def step(dp, i):
            curr_idx = perm[i]
            prev_idx = perm[i-1]
            s_curr = segments[curr_idx]
            s_prev = segments[prev_idx]
            t_curr = print_times[curr_idx]
            
            # dp[0] is time ending at s_prev[1], dp[1] is time ending at s_prev[0]
            # New dp[0] (ending at s_curr[1]):
            # 1. From s_prev[1] -> s_curr[0] -> s_curr[1]
            # 2. From s_prev[0] -> s_curr[0] -> s_curr[1]
            time_to_C = min(
                dp[0] + dist(s_prev[1], s_curr[0]) / S,
                dp[1] + dist(s_prev[0], s_curr[0]) / S
            ) + t_curr
            
            # New dp[1] (ending at s_curr[0]):
            # 1. From s_prev[1] -> s_curr[1] -> s_curr[0]
            # 2. From s_prev[0] -> s_curr[1] -> s_curr[0]
            time_to_A = min(
                dp[0] + dist(s_prev[1], s_curr[1]) / S,
                dp[1] + dist(s_prev[0], s_curr[1]) / S
            ) + t_curr
            
            return (time_to_C, time_to_A)

        final_dp = reduce(step, range(1, N), start_dp)
        return min(final_dp)

    # Evaluate all permutations and find the minimum
    ans = min(map(calculate_path_time, permutations(range(N))))
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()