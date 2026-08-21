import itertools
import math
import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute printing time for each segment
    print_times = [dist(s[0], s[1]) / T for s in segments]
    total_print_time = sum(print_times)

    # We need to find the minimum travel time between segments.
    # There are N! permutations of segments and 2^N ways to choose direction.
    # Since N <= 6, N! * 2^N is at most 720 * 64 = 46,080, which is small.
    
    # Generate all permutations of segment indices
    perms = itertools.permutations(range(N))
    # Generate all possible direction choices (0: start->end, 1: end->start)
    dirs = itertools.product([0, 1], repeat=N)

    def calculate_travel_time(p, d):
        # p: permutation of indices, d: tuple of directions
        # Starting point is (0, 0)
        
        # Helper to get start and end points of a segment based on direction
        get_pts = lambda i: (segments[i][0], segments[i][1]) if d[i] == 0 else (segments[i][1], segments[i][0])
        
        # We need the sequence of points: Start0, End0, Start1, End1, ...
        # The travel time is dist(Origin, Start0)/S + dist(End0, Start1)/S + ...
        
        # Create the sequence of points
        pts = [get_pts(i) for i in p]
        
        # The travel legs are:
        # 1. (0,0) to pts[0][0]
        # 2. pts[i][1] to pts[i+1][0] for i in 0..N-2
        
        first_leg = dist((0, 0), pts[0][0]) / S
        other_legs = sum(dist(pts[i][1], pts[i+1][0]) / S for i in range(N - 1))
        
        return first_leg + other_legs

    # To optimize, we can iterate through permutations and for each, 
    # find the best directions. However, the direction of segment i 
    # depends on the end of i-1 and the start of i+1.
    # This looks like a small DP or just brute force.
    
    # Let's use a more efficient approach for directions given a permutation:
    # dp[i][dir] = min time to finish segment i ending in direction 'dir'
    def get_min_travel_for_perm(p):
        # dp[dir] stores the min travel time to reach the end of the current segment
        # dir 0: segment printed start->end, dir 1: end->start
        
        # Initial step
        s0, e0 = segments[p[0]]
        dp = [
            dist((0, 0), s0) / S, # Dir 0: (0,0) -> s0, then print to e0
            dist((0, 0), e0) / S  # Dir 1: (0,0) -> e0, then print to s0
        ]
        
        # Since we can't use loops, we use functools.reduce to simulate DP
        from functools import reduce
        
        def step(acc, i):
            curr_s, curr_e = segments[p[i]]
            # New DP state:
            # next_dp[0] (end at curr_e): 
            #    min(acc[0] + dist(prev_e, curr_s)/S, acc[1] + dist(prev_s, curr_s)/S)
            # next_dp[1] (end at curr_s):
            #    min(acc[0] + dist(prev_e, curr_e)/S, acc[1] + dist(prev_s, curr_e)/S)
            
            # To get prev_s and prev_e, we need the segment from the previous step
            prev_idx = p[i-1]
            prev_s, prev_e = segments[prev_idx]
            
            return [
                min(acc[0] + dist(prev_e, curr_s) / S, acc[1] + dist(prev_s, curr_s) / S),
                min(acc[0] + dist(prev_e, curr_e) / S, acc[1] + dist(prev_s, curr_e) / S)
            ]
        
        final_dp = reduce(step, range(1, N), dp)
        return min(final_dp)

    # Use map and min to find the overall minimum travel time across all permutations
    min_travel = min(map(get_min_travel_for_perm, itertools.permutations(range(N))))
    
    print(f"{total_print_time + min_travel:.20f}")

if __name__ == "__main__":
    solve()