import sys
from itertools import permutations
import math

def solve():
    # Read input and parse N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples: ((x1, y1), (x2, y2))
    segments = [
        (
            (int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
            (int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        )
        for i in range(N)
    ]
    
    # Precompute lengths of each segment (time taken to print)
    # length / T
    seg_times = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2) / T
        for s in segments
    ]
    
    # We need to visit every segment. For each segment, we can start at either end.
    # There are N! permutations of segments and 2^N ways to choose directions.
    # Since N is small (<= 6), we can iterate through all permutations and 
    # use dynamic programming or recursion to find the best direction for each.
    
    # For a fixed permutation of segments, the state can be (index_of_segment, end_point_used).
    # However, with N=6, we can simply iterate all 2^N direction combinations 
    # inside the permutation loop, or use a small DP.
    
    # Let's use a helper to calculate distance between two points
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # For a fixed order of segments, we want to minimize travel time.
    # dp[i][0] = min time to finish i-th segment ending at endpoint 0
    # dp[i][1] = min time to finish i-th segment ending at endpoint 1
    
    def get_min_time(p):
        # p is a permutation of indices 0...N-1
        # Initial state: distance from (0,0) to start of first segment
        # Segment p[0] has endpoints A and B.
        # If we end at A, we started at B. If we end at B, we started at A.
        
        # s_idx = p[0]
        # dp0: min time ending at segments[s_idx][0]
        # dp1: min time ending at segments[s_idx][1]
        
        # To end at index 0, we must have traveled (0,0) -> point 1 -> point 0
        # To end at index 1, we must have traveled (0,0) -> point 0 -> point 1
        
        s0 = segments[p[0]][0]
        s1 = segments[p[0]][1]
        
        # Time to print the segment is constant regardless of direction
        print_time0 = seg_times[p[0]]
        
        # Initial DP values
        # dp[0] is time ending at point 0, dp[1] is time ending at point 1
        dp0 = (dist((0, 0), s1) / S) + print_time0
        dp1 = (dist((0, 0), s0) / S) + print_time0
        
        current_state = (dp0, dp1)
        
        for i in range(1, N):
            prev_p = p[i-1]
            curr_p = p[i]
            
            prev_pts = segments[prev_p]
            curr_pts = segments[curr_p]
            
            # Time to print current segment
            t_print = seg_times[curr_p]
            
            # New dp0: ending at curr_pts[0] (so started at curr_pts[1])
            # Option 1: prev ended at prev_pts[0] -> move to curr_pts[1] -> print to curr_pts[0]
            # Option 2: prev ended at prev_pts[1] -> move to curr_pts[1] -> print to curr_pts[0]
            res0 = min(
                current_state[0] + dist(prev_pts[0], curr_pts[1]) / S,
                current_state[1] + dist(prev_pts[1], curr_pts[1]) / S
            ) + t_print
            
            # New dp1: ending at curr_pts[1] (so started at curr_pts[0])
            res1 = min(
                current_state[0] + dist(prev_pts[0], curr_pts[0]) / S,
                current_state[1] + dist(prev_pts[1], curr_pts[0]) / S
            ) + t_print
            
            current_state = (res0, res1)
            
        return min(current_state)

    # Try all permutations of segment indices
    all_perms = permutations(range(N))
    ans = min(get_min_time(p) for p in all_perms)
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()