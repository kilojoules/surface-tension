import sys
import math
from itertools import permutations

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    
    segments = []
    ptr = 3
    for _ in range(N):
        ax = int(input_data[ptr])
        ay = int(input_data[ptr+1])
        cx = int(input_data[ptr+2])
        cy = int(input_data[ptr+3])
        segments.append(((ax, ay), (cx, cy)))
        ptr += 4

    # Precompute lengths of segments and time to print them
    seg_lengths = []
    for p1, p2 in segments:
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        seg_lengths.append(dist)

    # We need to visit all segments. For each segment, we can start at either endpoint.
    # Since N is small (N <= 6), we can iterate through all permutations of segments
    # and all possible directions (start at p1 or p2).
    
    min_total_time = float('inf')
    
    # Permutations of segments
    for p in permutations(range(N)):
        # There are 2^N ways to choose the direction for each segment
        # We can use recursion or bitmask to handle the directions.
        # For a fixed permutation, we want to find the best directions.
        # dp[i][end_point_index] = min time to print first i segments ending at end_point_index (0 or 1)
        
        # dp[i][0]: finished i-th segment (in permutation p) and ended at p[i]'s point 0
        # dp[i][1]: finished i-th segment (in permutation p) and ended at p[i]'s point 1
        
        # Initialize DP for the first segment in the permutation
        idx0 = p[0]
        p1_0, p2_0 = segments[idx0]
        
        # Option 1: Start at p1_0, end at p2_0
        # Time = dist(origin, p1_0)/S + length/T
        d_start_p1 = math.sqrt(p1_0[0]**2 + p1_0[1]**2)
        dp0 = (d_start_p1 / S) + (seg_lengths[idx0] / T)
        
        # Option 2: Start at p2_0, end at p1_0
        d_start_p2 = math.sqrt(p2_0[0]**2 + p2_0[1]**2)
        dp1 = (d_start_p2 / S) + (seg_lengths[idx0] / T)
        
        current_dp = [dp0, dp1]
        
        for i in range(1, N):
            idx_prev = p[i-1]
            idx_curr = p[i]
            
            prev_p1, prev_p2 = segments[idx_prev]
            curr_p1, curr_p2 = segments[idx_curr]
            
            # next_dp[0] means we end at curr_p1 (so we started at curr_p2)
            # next_dp[1] means we end at curr_p2 (so we started at curr_p1)
            
            # To end at curr_p1, we must move from prev_end to curr_p2, then print to curr_p1
            # prev_end is either prev_p1 (if current_dp[0]) or prev_p2 (if current_dp[1])
            
            # Calculate time to end at curr_p1
            dist_p1_to_p2 = seg_lengths[idx_curr]
            
            # From prev_p1 to curr_p2
            d0_to_p2 = math.sqrt((prev_p1[0]-curr_p2[0])**2 + (prev_p1[1]-curr_p2[1])**2)
            # From prev_p2 to curr_p2
            d1_to_p2 = math.sqrt((prev_p2[0]-curr_p2[0])**2 + (prev_p2[1]-curr_p2[1])**2)
            
            cost_to_p1 = min(current_dp[0] + d0_to_p2/S, current_dp[1] + d1_to_p2/S) + dist_p1_to_p2/T
            
            # Calculate time to end at curr_p2
            # From prev_p1 to curr_p1
            d0_to_p1 = math.sqrt((prev_p1[0]-curr_p1[0])**2 + (prev_p1[1]-curr_p1[1])**2)
            # From prev_p2 to curr_p1
            d1_to_p1 = math.sqrt((prev_p2[0]-curr_p1[0])**2 + (prev_p2[1]-curr_p1[1])**2)
            
            cost_to_p2 = min(current_dp[0] + d0_to_p1/S, current_dp[1] + d1_to_p1/S) + dist_p1_to_p2/T
            
            current_dp = [cost_to_p1, cost_to_p2]
            
        min_total_time = min(min_total_time, min(current_dp))

    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()