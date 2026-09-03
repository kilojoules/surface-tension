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
    # Time to print segment i is length / T
    seg_times = []
    for p1, p2 in segments:
        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        seg_times.append(dist / T)

    # We need to visit all N segments. For each segment, we can start at either end.
    # There are N! permutations of segments and 2^N choices of directions.
    # Since N is small (<= 6), we can iterate through all permutations and use DP or recursion.
    
    # Let's use recursion with memoization or just iterate since N! * 2^N is small.
    # N=6: 720 * 64 = 46,080 iterations.
    
    min_total_time = float('inf')
    
    # Precompute distances between all endpoints
    # Endpoints: (0,0) is index 0, then segment i has endpoints 2*i-1 and 2*i
    coords = [(0.0, 0.0)]
    for p1, p2 in segments:
        coords.append(p1)
        coords.append(p2)
        
    def get_dist(i, j):
        p1 = coords[i]
        p2 = coords[j]
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Try all permutations of segments
    for p in permutations(range(N)):
        # For each permutation, we have 2^N choices of direction.
        # We can use DP to find the best direction for this specific permutation.
        # dp[i][0] = min time to finish segment p[i] ending at its 1st endpoint
        # dp[i][1] = min time to finish segment p[i] ending at its 2nd endpoint
        
        # Initial step: from (0,0) to segment p[0]
        # Segment p[0] endpoints are coords[2*p[0]+1] and coords[2*p[0]+2]
        
        # Endpoints of segment i are:
        # Start-point A: 2*i + 1
        # End-point B: 2*i + 2
        
        # If we end at A, we must have started at B.
        # Time = dist(current, B)/S + length(segment)/T
        
        # Base case for the first segment in permutation p
        seg0 = p[0]
        # Option 0: End at A (2*seg0+1), started at B (2*seg0+2)
        dp0 = (get_dist(0, 2*seg0+2) / S) + seg_times[seg0]
        # Option 1: End at B (2*seg0+2), started at A (2*seg0+1)
        dp1 = (get_dist(0, 2*seg0+1) / S) + seg_times[seg0]
        
        for i in range(1, N):
            curr_seg = p[i]
            prev_seg = p[i-1]
            
            # Current segment endpoints
            cA = 2*curr_seg + 1
            cB = 2*curr_seg + 2
            # Previous segment endpoints
            pA = 2*prev_seg + 1
            pB = 2*prev_seg + 2
            
            # To end at cA, we must start at cB.
            # We could have come from pA or pB.
            next_dp0 = min(dp0 + get_dist(pA, cB)/S, dp1 + get_dist(pB, cB)/S) + seg_times[curr_seg]
            
            # To end at cB, we must start at cA.
            next_dp1 = min(dp0 + get_dist(pA, cA)/S, dp1 + get_dist(pB, cA)/S) + seg_times[curr_seg]
            
            dp0, dp1 = next_dp0, next_dp1
            
        min_total_time = min(min_total_time, dp0, dp1)

    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()