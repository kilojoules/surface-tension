import sys
import math
from itertools import permutations, product

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into pairs of points
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]
    
    # Precompute lengths of each segment
    lengths = [
        math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        for p1, p2 in segments
    ]
    
    # Total time spent emitting the laser is constant regardless of order
    total_emit_time = sum(lengths) / T
    
    # We need to find the minimum travel time between segments
    # A state is defined by the sequence of segments and the direction (0 or 1)
    # for each segment (which endpoint is the start and which is the end).
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    # Generate all possible direction combinations (0: A->C, 1: C->A)
    all_dirs = product([0, 1], repeat=N)
    
    # To avoid loops, we use a generator expression inside min()
    # We calculate the travel time for a specific permutation and direction set.
    # The travel time depends on the distance from the previous end point to the next start point.
    
    def get_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # We need to evaluate all permutations and all direction combinations.
    # Since N is small (<= 6), N! * 2^N is at most 720 * 64 = 46080.
    
    # We use a helper to get the start and end points based on the direction bit
    # dir=0: start=seg[0], end=seg[1]; dir=1: start=seg[1], end=seg[0]
    
    # To calculate the total travel time for a given order and direction:
    # 1. Start at (0,0)
    # 2. Move to start of 1st segment
    # 3. Move from end of i-th to start of (i+1)-th
    
    # We use a list comprehension to evaluate the cost for each permutation and direction
    # and then take the minimum.
    
    # Because we cannot use loops, we use a nested generator.
    # For a fixed permutation 'p' and direction 'd':
    # Points visited: (0,0) -> Start(p0, d0) -> End(p0, d0) -> Start(p1, d1) -> ...
    
    # To calculate the sum of travel distances without a loop, 
    # we can create a list of points and use zip to find distances between them.
    
    # Since we can't use loops to build the point list, we use a list comprehension.
    
    def calculate_travel_time(p, d):
        # Create the sequence of points: [Start0, End0, Start1, End1, ...]
        # The travel time is distance((0,0), Start0) + distance(End0, Start1) + ...
        
        # Map the permutation and direction to actual coordinates
        coords = [
            (segments[p[i]][0] if d[i] == 0 else segments[p[i]][1],
             segments[p[i]][1] if d[i] == 0 else segments[p[i]][0])
            for i in range(N)
        ]
        
        # The points we move between (without laser) are:
        # (0,0) -> coords[0][0]
        # coords[0][1] -> coords[1][0]
        # ...
        # coords[N-1][0] -> coords[N-1][1] is the laser part (already handled)
        
        # Travel segments: 
        # 0: (0,0) to coords[0][0]
        # i: coords[i-1][1] to coords[i][0] for i in 1..N-1
        
        # We use a list comprehension to get all travel distances
        dists = [get_dist((0, 0), coords[0][0])] + [
            get_dist(coords[i-1][1], coords[i][0]) 
            for i in range(1, N)
        ]
        return sum(dists) / S

    # We need to iterate over all permutations and all direction combinations.
    # Since we can't use nested loops, we use a generator expression.
    # Note: all_dirs needs to be recreated or converted to a list because it's an iterator.
    
    # To avoid the iterator exhaustion problem and loops, 
    # we wrap the logic in a way that computes the minimum.
    
    # We pre-calculate all direction combinations as a list.
    dirs_list = list(product([0, 1], repeat=N))
    
    # The final result is the minimum travel time + the constant emit time.
    ans = min(
        calculate_travel_time(p, d) 
        for p in permutations(range(N)) 
        for d in dirs_list
    ) + total_emit_time
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()