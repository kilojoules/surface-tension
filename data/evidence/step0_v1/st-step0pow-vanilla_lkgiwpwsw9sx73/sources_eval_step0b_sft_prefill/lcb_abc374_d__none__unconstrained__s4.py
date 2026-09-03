import sys
import math
from itertools import permutations

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Store segments as pairs of points
    segments = []
    idx = 3
    for i in range(N):
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))
        idx += 4

    # Precompute lengths of segments
    # length[i] = distance between endpoints of segment i
    lengths = [math.sqrt((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2) for s in segments]
    
    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to find a permutation of segments and a choice of direction for each
    # Since N is very small (up to 6), we can iterate through all permutations (N!)
    # and all direction combinations (2^N).
    # Total complexity: N! * 2^N * N, which is 720 * 64 * 6 ≈ 276,480 operations.
    
    # To avoid loops and recursion, we use list comprehensions and min()
    # We represent the state as (current_point, remaining_segments_mask)
    # However, given the constraints and the "no recursion" preference for some, 
    # we can use a bitmask DP or simply iterate through permutations.
    
    # Let's use a DP approach with bitmask to find the minimum time.
    # dp[mask][last_point_idx] = min_time
    # mask: bitmask of visited segments
    # last_point_idx: index of the point where the laser is (0 to 2N-1)
    # Point 0: (0,0), Points 1,2: segment 0, Points 3,4: segment 1...
    
    points = [(0, 0)] + [p for s in segments for p in s]
    
    # Initialize DP table with infinity
    # dp[mask][point_idx]
    # mask is from 0 to 2^N - 1, point_idx is from 0 to 2*N
    dp = [[float('inf')] * (2 * N + 1) for _ in range(1 << N)]
    
    # Base case: start at (0,0)
    dp[0][0] = 0.0
    
    # Iterate through all masks
    for mask in range(1 << N):
        for curr_p in range(2 * N + 1):
            if dp[mask][curr_p] == float('inf'):
                continue
            
            # Try to visit an unvisited segment i
            for i in range(N):
                if not (mask & (1 << i)):
                    # Segment i has endpoints p1 and p2
                    p1_idx = 2 * i + 1
                    p2_idx = 2 * i + 2
                    
                    # Option 1: Move to p1, print to p2
                    # Time = dist(curr, p1)/S + length[i]/T
                    time1 = dp[mask][curr_p] + dist(points[curr_p], points[p1_idx]) / S + lengths[i] / T
                    new_mask = mask | (1 << i)
                    if time1 < dp[new_mask][p2_idx]:
                        dp[new_mask][p2_idx] = time1
                        
                    # Option 2: Move to p2, print to p1
                    # Time = dist(curr, p2)/S + length[i]/T
                    time2 = dp[mask][curr_p] + dist(points[curr_p], points[p2_idx]) / S + lengths[i] / T
                    if time2 < dp[new_mask][p1_idx]:
                        dp[new_mask][p1_idx] = time2

    # The answer is the minimum time in dp[(1 << N) - 1]
    print(f"{min(dp[(1 << N) - 1]):.20f}")

if __name__ == "__main__":
    solve()