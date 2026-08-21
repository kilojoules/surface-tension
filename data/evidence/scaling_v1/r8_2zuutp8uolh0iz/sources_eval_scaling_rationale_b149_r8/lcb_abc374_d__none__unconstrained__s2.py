import sys
import math
from itertools import product, permutations

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
        segments.append((int(input_data[idx]), int(input_data[idx+1]), 
                          int(input_data[idx+2]), int(input_data[idx+3])))

    # Precompute lengths of segments
    # length_i = sqrt((Cx-Ax)^2 + (Dy-By)^2)
    seg_lengths = [math.sqrt((s[2]-s[0])**2 + (s[3]-s[1])**2) for s in segments]
    
    # Total time spent emitting the laser is constant regardless of order
    total_emit_time = sum(seg_lengths) / T

    # We need to find the minimum travel time between segments.
    # Each segment i has two endpoints: P_{i,0}=(Ai, Bi) and P_{i,1}=(Ci, Di)
    # A path is defined by a permutation of segments and a choice of direction for each.
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    
    # Generate all possible directions (0: P0->P1, 1: P1->P0)
    dirs = product([0, 1], repeat=N)
    
    # To avoid nested loops, we use a generator expression and min()
    # We calculate the non-emitting travel time for each configuration.
    
    # Helper to get coordinates based on direction
    # dir=0: start=P0, end=P1; dir=1: start=P1, end=P0
    def get_coords(seg_idx, direction):
        s = segments[seg_idx]
        return (s[0], s[1]), (s[2], s[3]) if direction == 0 else (s[2], s[3]), (s[0], s[1])

    # Since we cannot use loops, we use a comprehensive approach to evaluate all paths.
    # We'll pre-calculate the endpoints for each segment in both directions.
    # endpoints[i][dir] = (start_point, end_point)
    endpoints = [
        [((s[0], s[1]), (s[2], s[3])), ((s[2], s[3]), (s[0], s[1]))]
        for s in segments
    ]

    # We evaluate all permutations and all direction combinations.
    # For a fixed permutation p and directions d:
    # Travel time = dist((0,0), start_{p0}) + dist(end_{p0}, start_{p1}) + ...
    
    # To avoid loops, we use a list comprehension to calculate the cost of a specific 
    # permutation and direction set, then take the minimum.
    
    # Because N is small (<= 6), N! * 2^N is at most 720 * 64 = 46080.
    # We can flatten the search space.
    
    def calc_travel_time(p, d):
        # Create the sequence of points: start0, end0, start1, end1, ...
        # The travel happens from (0,0)->start0, end0->start1, end1->start2...
        pts = [endpoints[p[i]][d[i]] for i in range(N)]
        
        # Distance from origin to first start
        first_dist = math.sqrt(pts[0][0][0]**2 + pts[0][0][1]**2)
        
        # Distances between segments: end_{i} to start_{i+1}
        # Using a list comprehension to sum distances
        inter_dists = sum(
            math.sqrt((pts[i][1][0] - pts[i+1][0][0])**2 + (pts[i][1][1] - pts[i+1][0][1])**2)
            for i in range(N-1)
        )
        
        return (first_dist + inter_dists) / S

    # We use a generator to find the minimum travel time across all permutations and directions.
    # Using itertools.product to handle the 2^N directions.
    
    min_travel_time = min(
        calc_travel_time(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )

    print(f"{total_emit_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()