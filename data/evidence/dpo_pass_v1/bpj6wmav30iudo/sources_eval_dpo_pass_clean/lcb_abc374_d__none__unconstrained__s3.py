import sys
import math
from itertools import permutations, product

def solve():
    # Read input using map and split
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, S, T and the line segments
    # N is index 0, S is index 1, T is index 2
    # Segments start from index 3
    N, S, T = map(int, input_data[:3])
    
    # Create a list of tuples representing the segments: ((Ax, Ay), (Cx, Cy))
    segments = [
        ((int(input_data[i]), int(input_data[i+1])), 
         (int(input_data[i+2]), int(input_data[i+3])))
        for i in range(3, len(input_data), 4)
    ]

    # Precompute the length of each segment to avoid redundant calculations
    # Time to print segment i is length / T
    seg_times = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2) / T
        for s in segments
    ]

    # Function to calculate distance between two points
    # Used inside a lambda to avoid 'def'
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # We need to check all permutations of segments (N!)
    # For each segment, we can start at either endpoint (2^N)
    # Total complexity: N! * 2^N * N. With N=6, this is 720 * 64 * 6 ≈ 276,480 operations.
    
    # Generate all permutations of indices 0 to N-1
    all_orders = permutations(range(N))
    
    # Generate all possible direction choices (0: A->C, 1: C->A)
    all_directions = product([0, 1], repeat=N)

    # Use a generator expression to calculate total time for every possible strategy
    # For a fixed order and fixed directions:
    # Current position starts at (0,0)
    # For each segment:
    #   1. Move from current pos to start point (dist / S)
    #   2. Move from start point to end point (dist / T)
    #   3. Update current pos to end point
    
    # Since we cannot use loops, we use a list comprehension to evaluate 
    # the path cost. To handle the "current position" state without a loop, 
    # we can pre-calculate the endpoints based on the direction choice.
    
    # Let's define the endpoints for each segment based on the direction bit
    # If bit is 0: start = A, end = C. If bit is 1: start = C, end = A.
    
    # To calculate the total travel time without a loop, we can use 
    # a list comprehension to get the sequence of points visited.
    
    # For a specific permutation 'p' and direction tuple 'd':
    # Points: P0(0,0) -> P1_start -> P1_end -> P2_start -> P2_end ...
    
    # We can use map and lambda to calculate the cost of a single configuration
    calc_cost = lambda p, d: (
        sum([
            # Printing time for each segment in the permutation
            seg_times[p[i]] for i in range(N)
        ]) +
        sum([
            # Travel time from end of prev segment to start of next
            # P_prev_end is the endpoint of segment p[i-1] with direction d[p[i-1]]
            # P_curr_start is the startpoint of segment p[i] with direction d[p[i]]
            dist(
                (segments[p[i-1]][(d[p[i-1]]+1)%2] if i > 0 else (0,0)),
                (segments[p[i]][d[p[i]]])
            ) / S
            for i in range(N)
        ])
    )

    # We need to iterate through all permutations and all direction combinations.
    # Since we can't use nested loops, we use a list comprehension 
    # and find the minimum.
    # Note: 'd' is a tuple of length N where d[i] is the choice for segment i.
    
    # To avoid the O(N! * 2^N) being too slow in a single line, 
    # we structure it carefully.
    
    # We can iterate over permutations, and for each permutation, 
    # the choice of direction for segment i only depends on the 
    # endpoint of segment i-1 and the startpoint of segment i.
    # Actually, for a fixed order, the optimal direction for segment i 
    # depends on the previous segment's end. This looks like DP, 
    # but with N=6, brute force is fine.
    
    # Correcting the logic: d[i] should be the direction for the i-th segment in the permutation.
    # Let's use a simpler approach: for a fixed permutation p, 
    # we try all 2^N combinations of start/end.
    
    ans = min(
        calc_cost(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )
    
    # The calc_cost above used d[p[i]], meaning d is mapped to the segment index.
    # Let's refine it to be absolutely sure.
    # For a permutation p, let's say we decide for each step i whether to 
    # go from A to C or C to A.
    
    # Let's redefine:
    # p: permutation of 0..N-1
    # d: tuple of 0/1 of length N, where d[i] is the direction for the i-th segment in the order p.
    
    # Revised calc_cost:
    # p = (2, 0, 1) -> segments[p[0]], segments[p[1]], segments[p[2]]
    # d = (0, 1, 0) -> seg p[0] (A->C), seg p[1] (C->A), seg p[2] (A->C)
    
    # Let's use a helper to get points:
    # get_pt(seg_idx, is_end) -> returns point
    # if d[i] == 0: start is segments[p[i]][0], end is segments[p[i]][1]
    # if d[i] == 1: start is segments[p[i]][1], end is segments[p[i]][0]
    
    # Total Time = Sum(Printing Times) + Sum(Travel Times)
    # Printing Times is constant for any valid strategy: sum(seg_times)
    
    # Travel Time = dist((0,0), start_0)/S + dist(end_0, start_1)/S + ...
    
    # Let',s just use the most direct comprehension:
    # result = min(
    #    sum(seg_times) + 
    #    sum(dist(
    #        (0,0) if i == 0 else (segments[p[i-1]][(d[i-1]+1)%2] if d[i-1]==0 else segments[p[i-1]][0]), 
    #        (segments[p[i]][0] if d[i]==0 else segments[p[i]][1])
    #    ) / S for i in range(N))
    #    for p in permutations(range(N))
    #    for d in product([0, 1], repeat=N)
    # )
    # Wait, the logic (segments[p[i-1]][(d[i-1]+1)%2] if d[i-1]==0 else segments[p[i-1]][0]) is just 
    # segments[p[i-1]][1 if d[i-1]==0 else 0].
    
    # Let',s finalize the formula:
    # For a permutation p and direction tuple d:
    # Segment i is p[i].
    # If d[i] == 0: start = segments[p[i]][0], end = segments[p[i]][1]
    # If d[i] == 1: start = segments[p[i]][1], end = segments[p[i]][0]
    
    # Travel distance:
    # i=0: dist((0,0), start_0)
    # i=1..N-1: dist(end_{i-1}, start_i)
    
    # Printing time:
    # sum(dist(segments[i][0], segments[i][1]) / T for i in range(N))
    
    # Final implementation using the logic above:
    print(f"{min(
        sum(seg_times) + sum(
            dist(
                (0, 0) if i == 0 else (segments[p[i-1]][1 if d[i-1] == 0 else 0]),
                (segments[p[i]][0] if d[i] == 0 else segments[p[i]][1])
            ) / S
            for i in range(N)
        )
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    ):.20f}")

if __name__ == "__main__":
    solve()