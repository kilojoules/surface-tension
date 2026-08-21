import sys
from itertools import permutations

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into pairs of endpoints
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    # Precompute lengths of segments (time spent printing)
    # Length L takes L/T seconds
    seg_lengths = [
        ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 
        for p1, p2 in segments
    ]
    total_print_time = sum(seg_lengths) / T

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    # We need to find the optimal order of segments and the direction of each segment.
    # There are N! permutations of segments and 2^N choices of directions.
    # For a fixed permutation and fixed directions, the travel time is:
    # dist(start, seg1_start)/S + dist(seg1_end, seg2_start)/S + ...
    
    # To optimize, for a fixed permutation, we can use DP or recursion to find the best directions.
    # However, N is very small (N <= 6), so we can iterate through all 2^N directions.
    # But wait, for a fixed permutation, the choice of direction for segment i only depends on 
    # the end of segment i-1 and the start of segment i+1.
    
    # Let's define a function that calculates the minimum travel time for a fixed permutation.
    # For a fixed permutation P, let f(i, end_point_of_i) be the min travel time to complete i segments.
    # Since N is tiny, we can just use a list comprehension to evaluate all 2^N direction combinations.
    
    # We use a generator expression inside min() to find the best permutation and direction set.
    # directions: 0 means (A, B) -> (C, D), 1 means (C, D) -> (A, B)
    
    # To avoid loops, we use a recursive-like structure via a helper or just map/comprehensions.
    # Given the constraints on "no loops", we can use a nested comprehension.
    # For each permutation, we evaluate all 2^N direction combinations.
    
    # Let's pre-calculate the endpoints for each segment in both directions.
    # endpoints[i][0] = (start, end) if dir=0, endpoints[i][1] = (end, start) if dir=1
    endpoints = [ [(s[0], s[1]), (s[1], s[0])] for s in segments ]

    # We want to minimize:
    # dist((0,0), p[0].start)/S + dist(p[0].end, p[1].start)/S + ... + dist(p[N-2].end, p[N-1].start)/S
    
    # We can use a list comprehension to generate all binary strings of length N for directions.
    # But we can't use loops to generate them. We can use a recursive function.
    
    def get_directions(n):
        if n == 0: return [[]]
        return [d + [0] for d in get_directions(n-1)] + [d + [1] for d in get_directions(n-1)]

    all_dirs = get_directions(N)
    
    # The total time is total_print_time + min(travel_times)
    # travel_time = sum(dist(current_pos, next_start)) / S
    
    # For a fixed permutation 'p' and fixed directions 'd':
    # points = [endpoints[p[i]][d[i]] for i in range(N)]
    # travel_dist = dist((0,0), points[0][0]) + sum(dist(points[i][1], points[i+1][0]) for i in range(N-1))
    
    # We use a nested comprehension:
    # min(
    #   sum(
    #     dist(
    #       (0,0) if i == 0 else endpoints[p[i-1]][d[i-1]][1],
    #       endpoints[p[i]][d[i]][0]
    #     )
    #     for i in range(N)
    #   )
    #   for p in permutations(range(N))
    #   for d in all_dirs
    # ) / S
    
    # Since we can't use 'for' in a loop but can in comprehensions:
    ans = total_print_time + min(
        sum(
            dist(
                (0, 0) if i == 0 else endpoints[p[i-1]][d[i-1]][1],
                endpoints[p[i]][d[i]][0]
            )
            for i in range(N)
        )
        for p in permutations(range(N))
        for d in all_dirs
    ) / S

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()