import sys
from itertools import permutations
import math

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map input to variables
    # N S T, then A1 B1 C1 D1 ...
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    coords = list(map(int, input_data[3:]))
    
    # Group coordinates into segments: (x1, y1, x2, y2)
    segments = [
        (coords[i*4], coords[i*4+1], coords[i*4+2], coords[i*4+3]) 
        for i in range(N)
    ]
    
    # Precompute lengths of segments
    lengths = [
        math.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2) 
        for s in segments
    ]
    
    # The total time spent emitting the laser is constant regardless of order
    total_emit_time = sum(lengths) / T
    
    # We need to find the minimum time spent moving without emitting
    # The state is defined by the order of segments and the direction of each segment
    # Since N is small (up to 6), we can iterate through all permutations (N!) 
    # and all direction combinations (2^N).
    
    # Helper to get endpoints of a segment based on direction (0 or 1)
    # dir 0: p1 -> p2, dir 1: p2 -> p1
    def get_endpoints(seg_idx, direction):
        s = segments[seg_idx]
        p1 = (s[0], s[1])
        p2 = (s[2], s[3])
        return (p1, p2) if direction == 0 else (p2, p1)

    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # For each permutation, we want to find the min travel time.
    # We can use DP or recursion, but since N is only 6, 
    # we can just iterate through all 2^N direction combinations.
    # To optimize: for a fixed permutation, the choice of direction for segment i
    # only depends on the end of segment i-1 and affects the start of segment i+1.
    
    # Let f(i, end_point_idx) be the min travel time after printing i segments,
    # where end_point_idx is 0 or 1 (representing which endpoint of the i-th segment we ended at).
    
    def calculate_min_travel(perm):
        # dp[direction] = min travel time to reach the end of the current segment
        # direction 0: ended at p2, direction 1: ended at p1
        
        # Initial step: from (0,0) to start of first segment
        p1_0, p2_0 = get_endpoints(perm[0], 0)
        p1_1, p2_1 = get_endpoints(perm[0], 1)
        
        # dp[0]: we printed p1_0 -> p2_0. Travel was (0,0) -> p1_0
        # dp[1]: we printed p1_1 -> p2_1. Travel was (0,0) -> p1_1
        dp = [dist((0, 0), p1_0) / S, dist((0, 0), p1_1) / S]
        
        for i in range(1, N):
            curr_perm_idx = perm[i]
            # We need to calculate new_dp[0] and new_dp[1]
            # To get new_dp[0]: we end at p2 of curr. We could have come from dp[0] or dp[1]
            # p_prev_end is the point where we finished the previous segment.
            
            # Endpoints of previous segment
            prev_p1, prev_p2 = get_endpoints(perm[i-1], 0)
            # Endpoints of current segment
            curr_p1, curr_p2 = get_endpoints(curr_perm_idx, 0)
            
            # Option 0: current segment is curr_p1 -> curr_p2
            # Travel from prev_p2 (if dp[0]) or prev_p1 (if dp[1]) to curr_p1
            cost0 = min(dp[0] + dist(prev_p2, curr_p1) / S, 
                        dp[1] + dist(prev_p1, curr_p1) / S)
            
            # Option 1: current segment is curr_p2 -> curr_p1
            # Travel from prev_p2 (if dp[0]) or prev_p1 (if dp[1]) to curr_p2
            cost1 = min(dp[0] + dist(prev_p2, curr_p2) / S, 
                        dp[1] + dist(prev_p1, curr_p2) / S)
            
            dp = [cost0, cost1]
            
        return min(dp)

    # Find the minimum travel time across all permutations
    min_travel_time = min(calculate_min_travel(p) for p in all_perms)
    
    # Final answer is total emission time + minimum travel time
    print(f"{total_emit_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()