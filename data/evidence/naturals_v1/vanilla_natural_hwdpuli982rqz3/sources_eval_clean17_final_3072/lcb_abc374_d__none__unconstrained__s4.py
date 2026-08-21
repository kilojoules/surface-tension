import sys
import math
from itertools import permutations, product

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Store line segments as pairs of points
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]
    
    # Precompute lengths of each segment
    # length = sqrt((x2-x1)^2 + (y2-y1)^2)
    # time_to_print = length / T
    seg_lengths = [
        math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
        for p1, p2 in segments
    ]
    
    # Total printing time is constant regardless of order
    total_print_time = sum(seg_lengths) / T
    
    # We need to find the minimum travel time (non-emitting)
    # The state is defined by: (current_position, set_of_visited_segments)
    # Since N is very small (up to 6), we can iterate through all permutations of segments
    # and for each segment, decide which endpoint to start from.
    
    # Generate all permutations of segment indices (0 to N-1)
    all_orders = permutations(range(N))
    
    # For each segment, there are 2 choices of direction (start -> end or end -> start)
    # 2^N combinations. We can use product([0, 1], repeat=N)
    directions = product([0, 1], repeat=N)
    
    # To avoid nested loops and recursion, we use a generator/comprehension
    # We want to minimize: 
    # dist(Origin, Start1)/S + dist(End1, Start2)/S + ... + dist(End(N-1), StartN)/S
    
    def calculate_travel_dist(order, dirs):
        # order: tuple of segment indices
        # dirs: tuple of 0 or 1 (0 means p1->p2, 1 means p2->p1)
        
        # Get the sequence of points visited
        # For each i in order, we have a segment (p1, p2).
        # If dirs[i] == 0, we move to p1, then print to p2.
        # If dirs[i] == 1, we move to p2, then print to p1.
        
        points = [
            (segments[order[i]][dirs[i]], segments[order[i]][1 - dirs[i]]) 
            for i in range(N)
        ]
        
        # Distance from (0,0) to first start point
        d0 = math.sqrt(points[0][0][0]**2 + points[0][0][1]**2)
        
        # Distances between consecutive segments (end of prev to start of next)
        # Using sum with a generator to avoid explicit loops
        d_between = sum(
            math.sqrt((points[i][1][0] - points[i+1][0][0])**2 + 
                      (points[i][1][1] - points[i+1][0][1])**2)
            for i in range(N - 1)
        )
        
        return (d0 + d_between) / S

    # We need to find the minimum travel time across all permutations and directions.
    # Since we can't use for/while loops, we use min() with a generator.
    
    # Note: The 'dirs' needs to be aligned with the 'order'. 
    # The way product(repeat=N) works, the i-th element of 'dirs' corresponds to 
    # the i-th segment in the 'order' tuple.
    
    min_travel_time = min(
        calculate_travel_dist(order, dirs)
        for order in all_orders
        for dirs in product([0, 1], repeat=N)
    )
    
    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()