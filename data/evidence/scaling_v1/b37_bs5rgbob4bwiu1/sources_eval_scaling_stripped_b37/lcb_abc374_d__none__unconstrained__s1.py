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
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    # Precalculate the time to print each segment (length / T)
    # print_times[i] = length of segment i / T
    print_times = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2) / T
        for s in segments
    ]
    
    # Total printing time is constant regardless of order
    total_print_time = sum(print_times)

    # Function to calculate distance between two points
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # We need to visit every segment. For each segment, we can start at either end.
    # There are N! permutations of segments and 2^N choices of direction.
    # Since N is small (<= 6), we can iterate through all permutations.
    # For the direction, we can use a bitmask or recursion, but since we want 
    # the minimum travel time, we can use dynamic programming over the permutation.
    
    # For a fixed permutation P of indices 0...N-1:
    # let f(i, side) be the min travel time to finish segment P[i] ending at 'side' (0 or 1).
    # side 0: ended at P[i].endpoint1, side 1: ended at P[i].endpoint2.
    
    def get_min_travel_time(p):
        # Initial travel from (0,0) to the start of the first segment
        # Start at endpoint 0, end at endpoint 1
        # d0: distance (0,0) -> p[0].pt0
        # d1: distance (0,0) -> p[0].pt1
        pt0_start = segments[p[0]][0]
        pt1_start = segments[p[0]][1]
        
        # dp[0]: min travel time ending at endpoint 0 of current segment
        # dp[1]: min travel time ending at endpoint 1 of current segment
        # To end at 0, we must have started at 1.
        dp0 = dist((0, 0), pt1_start) / S
        dp1 = dist((0, 0), pt0_start) / S
        
        for i in range(1, N):
            prev_pt0 = segments[p[i-1]][0]
            prev_pt1 = segments[p[i-1]][1]
            curr_pt0 = segments[p[i]][0]
            curr_pt1 = segments[p[i]][1]
            
            # To end at curr_pt0, we must start at curr_pt1.
            # We could have come from prev_pt0 or prev_pt1.
            next_dp0 = min(
                dp0 + dist(prev_pt0, curr_pt1) / S,
                dp1 + dist(prev_pt1, curr_pt1) / S
            )
            # To end at curr_pt1, we must start at curr_pt0.
            next_dp1 = min(
                dp0 + dist(prev_pt0, curr_pt0) / S,
                dp1 + dist(prev_pt1, curr_pt0) / S
            )
            dp0, dp1 = next_dp0, next_dp1
            
        return min(dp0, dp1)

    # Try all permutations of segment indices
    all_perms = permutations(range(N))
    min_travel = min(get_min_travel_time(p) for p in all_perms)
    
    print(f"{total_print_time + min_travel:.20f}")

if __name__ == "__main__":
    solve()