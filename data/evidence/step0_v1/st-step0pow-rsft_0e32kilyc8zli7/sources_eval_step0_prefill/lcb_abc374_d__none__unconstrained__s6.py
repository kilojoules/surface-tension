import sys
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

    # Precompute lengths of segments
    seg_lengths = []
    for p1, p2 in segments:
        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        seg_lengths.append(dist)

    # The total time spent emitting the laser is constant regardless of order
    # Total emit time = sum(length_i / T)
    total_emit_time = sum(seg_lengths) / T

    # We need to find the minimum travel time (non-emitting)
    # The laser starts at (0, 0).
    # For each segment i, we can print it from p_{i,1} to p_{i,2} or vice versa.
    # This is a variation of the Traveling Salesperson Problem, but N is very small (N <= 6).
    # We can iterate through all permutations of segments and all possible directions.
    
    min_travel_dist = float('inf')
    
    # There are N! permutations of segments and 2^N combinations of directions.
    # Total combinations: 6! * 2^6 = 720 * 64 = 46,080. This is small enough.
    
    # Pre-calculate endpoints for each segment
    endpoints = [ (seg[0], seg[1]) for seg in segments ]
    
    # Try all permutations of segment indices
    for p in permutations(range(N)):
        # For each permutation, we use dynamic programming or recursion to find the best directions.
        # dp[i][side] = min distance to finish segment p[i] ending at side (0 or 1)
        # side 0: ended at endpoints[p[i]][0], side 1: ended at endpoints[p[i]][1]
        
        # Initial step: from (0,0) to start of first segment
        # If we end at side 0, we started at side 1.
        # Dist = dist((0,0), side 1)
        
        # Let's use a simple recursive approach with memoization or just iterate.
        # current_min_dists[side] stores the min distance to reach the end of the current segment.
        
        # First segment
        seg_idx = p[0]
        p1, p2 = endpoints[seg_idx]
        # Option 1: Start p2, end p1
        d0 = ((0 - p2[0])**2 + (0 - p2[1])**2)**0.5
        # Option 2: Start p1, end p2
        d1 = ((0 - p1[0])**2 + (0 - p1[1])**2)**0.5
        
        cur_dists = [d0, d1]
        
        for i in range(1, N):
            seg_idx = p[i]
            p1, p2 = endpoints[seg_idx]
            prev_p1, prev_p2 = endpoints[p[i-1]]
            
            # New d0: end at p1 (started at p2)
            # Can come from prev_p1 (cur_dists[0]) or prev_p2 (cur_dists[1])
            # Note: cur_dists[0] is the distance after finishing segment p[i-1] at prev_p1
            # Wait, the logic above is: cur_dists[0] is distance after finishing at endpoints[p[i-1]][0]
            
            # To end at p1, we must start at p2.
            # Travel from prev_p1 to p2 or prev_p2 to p2.
            dist_prev1_to_p2 = ((prev_p1[0] - p2[0])**2 + (prev_p1[1] - p2[1])**2)**0.5
            dist_prev2_to_p2 = ((prev_p2[0] - p2[0])**2 + (prev_p2[1] - p2[1])**2)**0.5
            
            # To end at p2, we must start at p1.
            dist_prev1_to_p1 = ((prev_p1[0] - p1[0])**2 + (prev_p1[1] - p1[1])**2)**0.5
            dist_prev2_to_p1 = ((prev_p2[0] - p1[0])**2 + (prev_p2[1] - p1[1])**2)**0.5
            
            next_d0 = min(cur_dists[0] + dist_prev1_to_p2, cur_dists[1] + dist_prev2_to_p2)
            next_d1 = min(cur_dists[0] + dist_prev1_to_p1, cur_dists[1] + dist_prev2_to_p1)
            cur_dists = [next_d0, next_d1]
            
        min_travel_dist = min(min_travel_dist, min(cur_dists))

    # Total time = (total_travel_dist / S) + total_emit_time
    print(f"{min_travel_dist / S + total_emit_time:.20f}")

if __name__ == "__main__":
    solve()