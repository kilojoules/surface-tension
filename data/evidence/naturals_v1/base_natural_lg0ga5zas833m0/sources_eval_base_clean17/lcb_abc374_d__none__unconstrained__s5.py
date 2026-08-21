import sys
import math
from itertools import permutations

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        a = int(input_data[idx])
        b = int(input_data[idx+1])
        c = int(input_data[idx+2])
        d = int(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precompute lengths of each segment
    seg_lengths = []
    for p1, p2 in segments:
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        seg_lengths.append(dist)

    # The total time spent emitting the laser is constant regardless of the order
    # Total emitting time = sum(length_i / T)
    total_emit_time = sum(seg_lengths) / T

    # We need to minimize the total time spent moving the laser without emitting.
    # The laser starts at (0, 0).
    # For each segment i, we can print from p1 to p2 or p2 to p1.
    # This is a variation of the TSP problem, but N is small (N <= 6).
    # We can iterate through all permutations of segments and all possible directions.

    min_move_dist = float('inf')
    
    # Permutations of segments
    for p in permutations(range(N)):
        # There are 2^N ways to choose the direction for each segment
        # Instead of iterating 2^N, we can use dynamic programming or recursion.
        # Since N is very small (6), we can use recursion or simply iterate 2^N.
        
        # dp[i][side] = min distance to finish segment i ending at side (0 or 1)
        # side 0: ended at p1, side 1: ended at p2
        
        # Initialize for the first segment in permutation p
        seg_idx = p[0]
        p1, p2 = segments[seg_idx]
        
        # Option 1: start at p1, end at p2
        # move from (0,0) to p1, then print p1 -> p2
        dist_start_p1 = math.sqrt(p1[0]**2 + p1[1]**2)
        # Option 2: start at p2, end at p1
        # move from (0,0) to p2, then print p2 -> p1
        dist_start_p2 = math.sqrt(p2[0]**2 + p2[1]**2)
        
        # dp[side] stores the min non-emitting distance to reach the end of the current segment
        dp = [dist_start_p1, dist_start_p2]
        
        for i in range(1, N):
            curr_seg_idx = p[i]
            prev_seg_idx = p[i-1]
            
            cp1, cp2 = segments[curr_seg_idx]
            pp1, pp2 = segments[prev_seg_idx]
            
            new_dp = [float('inf')] * 2
            
            # To end at cp2 (meaning we print cp1 -> cp2)
            # We could have come from pp1 (end of prev) or pp2 (end of prev)
            # If prev ended at pp1: move pp1 -> cp1
            # If prev ended at pp2: move pp2 -> cp1
            
            # Case 1: End at cp2 (print cp1 -> cp2)
            # From prev ending at pp1 (dp[0])
            d0 = math.sqrt((pp1[0] - cp1[0])**2 + (pp1[1] - cp1[1])**2)
            # From prev ending at pp2 (dp[1])
            d1 = math.sqrt((pp2[0] - cp1[0])**2 + (pp2[1] - cp1[1])**2)
            new_dp[0] = min(dp[0] + d0, dp[1] + d1)
            
            # Case 2: End at cp1 (print cp2 -> cp1)
            # From prev ending at pp1 (dp[0])
            d0 = math.sqrt((pp1[0] - cp2[0])**2 + (pp1[1] - cp2[1])**2)
            # From prev ending at pp2 (dp[1])
            d1 = math.sqrt((pp2[0] - cp2[0])**2 + (pp2[1] - cp2[1])**2)
            new_dp[1] = min(dp[0] + d0, dp[1] + d1)
            
            # Wait, the indices in dp are: 
            # dp[0] is distance when ending at p2 (printed p1->p2)
            # dp[1] is distance when ending at p1 (printed p2->p1)
            # Let's redefine:
            # dp[0]: ended at p2 (printed p1->p2)
            # dp[1]: ended at p1 (printed p2->p1)
            
            # To calculate new_dp[0] (end at cp2, print cp1->cp2):
            # from dp[0] (ended at pp2): move pp2 -> cp1
            # from dp[1] (ended at pp1): move pp1 -> cp1
            
            # Let's re-evaluate carefully.
            # prev_seg = (pp1, pp2)
            # current_seg = (cp1, cp2)
            # dp[0] is min dist ending at pp2
            # dp[1] is min dist ending at pp1
            
            # To end at cp2:
            # dist1 = dp[0] + dist(pp2, cp1)
            # dist2 = dp[1] + dist(pp1, cp1)
            # new_dp[0] = min(dist1, dist2)
            
            # To end at cp1:
            # dist1 = dp[0] + dist(pp2, cp2)
            # dist2 = dp[1] + dist(pp1, cp2)
            # new_dp[1] = min(dist1, dist2)
            
            # Correct logic:
            # Let's just use the distance from the previous endpoint to the start of the next segment.
            # If we ended at pp2, and we want to print cp1->cp2, we move pp2 -> cp1.
            # If we ended at pp1, and we want to print cp1->cp2, we move pp1 -> cp1.
            
            # Let's rewrite the inner loop.
            pass
            
        # Since the logic inside the loop was messy, I'll implement it cleanly.

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    segments = []
    idx = 3
    for _ in range(N):
        segments.append(((int(input_data[idx]), int(input_data[idx+1])), 
                         (int(input_data[idx+2]), int(input_data[idx+3]))))
        idx += 4

    total_emit_time = 0
    for p1, p2 in segments:
        total_emit_time += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / T

    min_move_time = float('inf')
    for p in permutations(range(N)):
        # dp[0]: min move dist ending at segment p[i]'s second endpoint (p2)
        # dp[1]: min move dist ending at segment p[i]'s first endpoint (p1)
        p0_idx = p[0]
        p1_start, p1_end = segments[p0_idx]
        dp = [
            math.sqrt(p1_start[0]**2 + p1_start[1]**2), # end at p1_end
            math.sqrt(p1_end[0]**2 + p1_end[1]**2)      # end at p1_start
        ]
        
        for i in range(1, N):
            prev_idx = p[i-1]
            curr_idx = p[i]
            pp1, pp2 = segments[prev_idx]
            cp1, cp2 = segments[curr_idx]
            
            # To end at cp2 (must start at cp1)
            d0 = dp[0] + math.sqrt((pp2[0]-cp1[0])**2 + (pp2[1]-cp1[1])**2)
            d1 = dp[1] + math.sqrt((pp1[0]-cp1[0])**2 + (pp1[1]-cp1[1])**2)
            new_dp0 = min(d0, d1)
            
            # To end at cp1 (must start at cp2)
            d0 = dp[0] + math.sqrt((pp2[0]-cp2[0])**2 + (pp2[1]-cp2[1])**2)
            d1 = dp[1] + math.sqrt((pp1[0]-cp2[0])**2 + (pp1[1]-cp2[1])**2)
            new_dp1 = min(d0, d1)
            
            dp = [new_dp0, new_dp1]
            
        min_move_time = min(min_move_time, min(dp) / S)

    print(f"{total_emit_time + min_move_time:.20f}")

if __name__ == "__main__":
    solve_final()