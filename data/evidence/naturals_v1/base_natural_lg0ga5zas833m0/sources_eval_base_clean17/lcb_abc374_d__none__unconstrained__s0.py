import sys
import math
from itertools import permutations

def solve():
    # Read N, S, T
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, s, t = map(int, line1)
    except ValueError:
        return

    segments = []
    for _ in range(n):
        segments.append(list(map(int, sys.stdin.readline().split())))

    # Precompute lengths of each segment
    # length = sqrt((x1-x2)^2 + (y1-y2)^2)
    # time to print = length / T
    seg_times = []
    endpoints = [] # List of (x, y) pairs for each segment
    for i in range(n):
        x1, y1, x2, y2 = segments[i]
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        seg_times.append(dist / t)
        endpoints.append(((x1, y1), (x2, y2)))

    # Since N is very small (N <= 6), we can iterate through all permutations
    # of the segments and for each segment, try both starting endpoints.
    # Total complexity: N! * 2^N
    # 6! * 2^6 = 720 * 64 = 46,080, which is well within limits.

    min_total_time = float('inf')
    
    # Permutation of segment indices
    for p in permutations(range(n)):
        # There are 2^N ways to choose the direction of printing for each segment
        # We can use a bitmask or recursion to try all 2^N combinations.
        # For a fixed permutation p, we can use DP or simple recursion.
        
        # dp[i][side] = min time to finish first i segments, ending at side of segment p[i]
        # side 0: endpoint 0, side 1: endpoint 1
        # Note: "ending at side" means the laser just finished printing the segment,
        # so the laser is at the 'other' end from where it started.
        
        # Let's define: 
        # For segment i, endpoints are E[i][0] and E[i][1].
        # If we print from 0 to 1, we end at E[i][1].
        # If we print from 1 to 0, we end at E[i][0].
        
        # Initial state: starting at (0, 0)
        # dp[0][0]: end at E[p[0]][0] (printed 1 -> 0)
        # dp[0][1]: end at E[p[0]][1] (printed 0 -> 1)
        
        e0 = endpoints[p[0]][0]
        e1 = endpoints[p[0]][1]
        
        # Time to print segment p[0] is constant regardless of direction
        t0 = seg_times[p[0]]
        
        # Starting from (0,0) to e1, then print e1 -> e0
        dp0 = (math.sqrt(e1[0]**2 + e1[1]**2) / s) + t0
        # Starting from (0,0) to e0, then print e0 -> e1
        dp1 = (math.sqrt(e0[0]**2 + e0[1]**2) / s) + t0
        
        current_dp = [dp0, dp1]
        
        for i in range(1, n):
            next_dp = [float('inf')] * 2
            idx = p[i]
            e_start0, e_start1 = endpoints[idx]
            t_print = seg_times[idx]
            
            # Previous endpoints
            prev_e0, prev_e1 = endpoints[p[i-1]]
            
            # To end at e_start0 (printed 1 -> 0)
            # From prev end 0 (which is e_start0 of p[i-1] if we ended at 0? No.)
            # Let's be careful: current_dp[0] is time ending at endpoints[p[i-1]][0]
            # current_dp[1] is time ending at endpoints[p[i-1]][1]
            
            prev_pos0 = endpoints[p[i-1]][0]
            prev_pos1 = endpoints[p[i-1]][1]
            
            # Try ending at e_start0 (printed e_start1 -> e_start0)
            # From prev_pos0
            d00 = math.sqrt((prev_pos0[0] - e_start1[0])**2 + (prev_pos0[1] - e_start1[1])**2) / s
            # From prev_pos1
            d10 = math.sqrt((prev_pos1[0] - e_start1[0])**2 + (prev_pos1[1] - e_start1[1])**2) / s
            next_dp[0] = min(current_dp[0] + d00, current_dp[1] + d10) + t_print
            
            # Try ending at e_start1 (printed e_start0 -> e_start1)
            # From prev_pos0
            d01 = math.sqrt((prev_pos0[0] - e_start0[0])**2 + (prev_pos0[1] - e_start0[1])**2) / s
            # From prev_pos1
            d11 = math.sqrt((prev_pos1[0] - e_start0[0])**2 + (prev_pos1[1] - e_start0[1])**2) / s
            next_dp[1] = min(current_dp[0] + d01, current_dp[1] + d11) + t_print
            
            current_dp = next_dp
            
        min_total_time = min(min_total_time, min(current_dp))

    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()