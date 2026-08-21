import sys
import math
from itertools import permutations

def solve():
    # Read N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        a = float(input_data[idx])
        b = float(input_data[idx+1])
        c = float(input_data[idx+2])
        d = float(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precompute lengths of each segment
    seg_lengths = []
    for p1, p2 in segments:
        length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        seg_lengths.append(length)

    # The number of segments N is very small (N <= 6).
    # We can iterate through all permutations of the segments and all possible 
    # directions (start point vs end point) for each segment.
    
    min_total_time = float('inf')
    
    # There are N! permutations of segments
    # For each segment, there are 2 choices of which endpoint to start from.
    # Total complexity: N! * 2^N * N, which for N=6 is 720 * 64 * 6 approx 276,480.
    # This is well within the time limit.
    
    # To optimize, we can use recursion or simply itertools.permutations
    for p in permutations(range(N)):
        # For a fixed order of segments, we can use DP or recursion to find the best directions.
        # dp[i][end_point_index] = min time to finish first i segments ending at endpoint index (0 or 1)
        
        # Initial state: from (0,0) to the first segment's chosen endpoint
        # We'll handle the first segment separately to initialize DP.
        
        # dp[endpoint_idx] stores the min time to reach that endpoint of the current segment
        # after printing it.
        
        # Segment p[0]
        p1, p2 = segments[p[0]]
        # Option 1: Start at p1, end at p2
        dist_to_p1 = math.sqrt(p1[0]**2 + p1[1]**2)
        time_p1_to_p2 = dist_to_p1 / S + seg_lengths[p[0]] / T
        
        # Option 2: Start at p2, end at p1
        dist_to_p2 = math.sqrt(p2[0]**2 + p2[1]**2)
        time_p2_to_p1 = dist_to_p2 / S + seg_lengths[p[0]] / T
        
        dp = [time_p1_to_p2, time_p2_to_p1] # dp[0] is ending at p2, dp[1] is ending at p1
        current_endpoints = [p2, p1]
        
        for i in range(1, N):
            seg_idx = p[i]
            p_start, p_end = segments[seg_idx]
            
            # Next DP state
            next_dp = [float('inf')] * 2
            
            # We are at current_endpoints[j], we want to print segment i.
            # We can start at p_start and end at p_end, or start at p_end and end at p_start.
            
            for j in range(2):
                curr_pos = current_endpoints[j]
                curr_time = dp[j]
                
                # Option A: Move to p_start, print to p_end
                dist_to_start = math.sqrt((curr_pos[0] - p_start[0])**2 + (curr_pos[1] - p_start[1])**2)
                next_dp[0] = min(next_dp[0], curr_time + dist_to_start / S + seg_lengths[seg_idx] / T)
                
                # Option B: Move to p_end, print to p_start
                dist_to_end = math.sqrt((curr_pos[0] - p_end[0])**2 + (curr_pos[1] - p_end[1])**2)
                next_dp[1] = min(next_dp[1], curr_time + dist_to_end / S + seg_lengths[seg_idx] / T)
            
            dp = next_dp
            current_endpoints = [p_end, p_start]
            
        min_total_time = min(min_total_time, min(dp))

    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()