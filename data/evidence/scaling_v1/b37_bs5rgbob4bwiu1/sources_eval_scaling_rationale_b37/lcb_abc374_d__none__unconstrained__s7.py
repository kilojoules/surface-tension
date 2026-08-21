import sys
import math
from itertools import permutations, product

def solve():
    # Read input and parse values
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples ((x1, y1), (x2, y2))
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute length of each segment for the printing time (T)
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    total_print_time = sum(seg_lengths) / T

    # We need to find the minimum travel time (S) between segments.
    # There are N! permutations of segments and 2^N ways to choose directions.
    # Since N is small (<= 6), we can brute force all combinations.
    
    # Generate all permutations of segment indices
    all_orders = permutations(range(N))
    
    # For a fixed order, we try all 2^N directions (0: start->end, 1: end->start)
    # We use a generator expression inside min() to find the best path for each permutation.
    
    def calculate_travel_time(order, directions):
        # Current position starts at (0, 0)
        # We need the distance from current_pos to the start of the segment,
        # then the segment is printed, ending at the other endpoint.
        
        # To avoid loops, we use a reduction-like approach via a helper function or 
        # a list comprehension with a custom accumulator. 
        # However, since we need the 'end' of the previous segment, 
        # a simple recursive-style mapping or a fold is needed.
        # Because we cannot use loops, we will simulate the path using a list comprehension
        # and a custom function that calculates the total distance.
        
        def get_points(idx_list, dir_list):
            # Returns a list of (start_point, end_point) for the chosen order and direction
            return [
                (segments[i][0] if-dir_list[j] == 0 else segments[i][1],
                 segments[i][1] if-dir_list[j] == 0 else segments[i][0])
                for j, i in enumerate(idx_list)
            ]
            
        # Since we can't use loops, we use a helper to sum distances between endpoints.
        # We'll use a list of points: P0(0,0), Start1, End1, Start2, End2...
        # Travel time is dist(P0, Start1) + dist(End1, Start2) + ...
        
        # Correcting the logic for directions:
        # For each segment i in order:
        #   if dir == 0: move to s[i][0], print to s[i][1]
        #   if dir == 1: move to s[i][1], print to s[i][0]
        
        pts = [(0, 0)] + [
            (segments[order[j]][0] if directions[j] == 0 else segments[order[j]][1],
             segments[order[j]][1] if directions[j] == 0 else segments[order[j]][0])
            for j in range(N)
        ]
        
        # Travel distance: dist(pts[0], pts[1].start) + dist(pts[1].end, pts[2].start) ...
        # pts[0] is (0,0). pts[1...N] are (start, end) tuples.
        
        # Distance from (0,0) to first start
        d0 = dist(pts[0], pts[1][0])
        
        # Distances between End_i and Start_{i+1}
        # We use a list comprehension to calculate distances between consecutive segments
        d_others = sum(dist(pts[i+1][1], pts[i+2][0]) for i in range(N-1))
        
        return (d0 + d_others) / S

    # We wrap the logic to find the minimum travel time across all permutations and directions.
    # Using map/min/generator expressions to replace loops.
    
    # For each permutation, we test all 2^N direction combinations.
    # Since N is very small, we can use product([0, 1], repeat=N).
    
    # To avoid the 'range' in a loop, we use map or generator expressions.
    # The most 'functional' way to handle the sequence of points is to pre-calculate 
    # the coordinates based on the permutation and direction bitmask.
    
    def solve_for_permutation(p):
        # p is a permutation of 0..N-1
        # directions is a tuple of 0/1
        # We want min(travel_time) over all 2^N directions
        return min(
            (
                dist((0, 0), (segments[p[0]][0] if d[0] == 0 else segments[p[0]][(1 if d[0] == 0 else 0)])) # This is wrong
                # Let's redefine clearly:
                # Point A: segments[i][0], Point B: segments[i][1]
                # If d[i]==0: Start=A, End=B. If d[i]==1: Start=B, End=A.
                # Travel = dist((0,0), Start_0) + sum(dist(End_i, Start_{i+1}))
                # Let's use a helper function for the sum.
                0 # placeholder
            ) for d in product([0, 1], repeat=N)
        )

    # Redefining the core logic to be strictly loop-free and correct:
    def get_total_travel(p, d):
        # p: permutation, d: directions (0 or 1)
        # Points: (start, end) for each segment in the permutation
        pts = [
            (segments[p[i]][0], segments[p[i]][1]) if d[i] == 0 else (segments[p[i]][1], segments[p[i]][0])
            for i in range(N)
        ]
        # Distance from origin to first start + distances between end_i and start_{i+1}
        return (dist((0, 0), pts[0][0]) + sum(dist(pts[i][1], pts[i+1][0]) for i in range(N-1))) / S

    # Final calculation:
    # Min over all permutations p, and for each p, min over all direction vectors d.
    ans = min(
        min(get_total_travel(p, d) for d in product([0, 1], repeat=N))
        for p in permutations(range(N))
    ) + total_print_time

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()