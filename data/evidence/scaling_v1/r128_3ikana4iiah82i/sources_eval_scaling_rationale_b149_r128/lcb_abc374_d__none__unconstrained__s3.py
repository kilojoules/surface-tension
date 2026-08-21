import sys
import math
from itertools import permutations, product

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse segments into pairs of coordinates
    segments = [
        ((int(input_data[3 + 4*i]), int(input_data[4 + 4*i])), 
         (int(input_data[5 + 4*i]), int(input_data[6 + 4*i])))
        for i in range(N)
    ]

    # Precompute lengths of segments for printing time
    # Printing time is always the same regardless of order or direction
    print_times = [math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / T 
                   for p1, p2 in segments]
    total_print_time = sum(print_times)

    # We need to find the minimum travel time between segments
    # There are N! permutations of segments and 2^N choices of directions
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # Generate all possible direction choices (0: p1->p2, 1: p2->p1)
    all_dirs = product([0, 1], repeat=N)

    def calculate_travel_time(perm, dirs):
        # Determine the sequence of points visited
        # Start at (0,0)
        points = [(0, 0)]
        
        # For each segment in the permutation, determine start and end points based on direction
        # We use a list comprehension to build the sequence of endpoints
        path = [
            (segments[perm[i]][dirs[i]], segments[perm[i]][1 - dirs[i]])
            for i in range(N)
        ]
        
        # The travel distance is the sum of distances between the end of segment i and start of segment i+1
        # Start point is (0,0), then we move to the start of the first segment, 
        # then from the end of the first to the start of the second, and so on.
        
        # Create a list of "jump" pairs: (current_pos, next_start_pos)
        # Jump 0: (0,0) -> path[0][0]
        # Jump i: path[i-1][1] -> path[i][0]
        jumps = [
            ((0, 0), path[0][0])
        ] + [
            (path[i-1][1], path[i][0]) 
            for i in range(1, N)
        ]
        
        # Calculate total travel distance / S
        return sum(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / S 
                   for p1, p2 in jumps)

    # We need to evaluate calculate_travel_time for all permutations and all direction combinations
    # Since N is small (<= 6), we can use a nested comprehension
    # Note: all_dirs is a generator, so we convert it to a list to reuse it for each permutation
    dir_list = list(all_dirs)
    
    min_travel_time = min(
        calculate_travel_time(p, d) 
        for p in all_perms 
        for d in dir_list
    )

    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()