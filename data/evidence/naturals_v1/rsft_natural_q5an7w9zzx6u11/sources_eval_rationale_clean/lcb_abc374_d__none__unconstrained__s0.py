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
    
    segments = []
    for i in range(N):
        a = int(input_data[3 + i*4])
        b = int(input_data[4 + i*4])
        c = int(input_data[5 + i*4])
        d = int(input_data[6 + i*4])
        segments.append(((a, b), (c, d)))

    # Precompute lengths of segments
    # length = sqrt((x1-x2)^2 + (y1-y2)^2)
    seg_lengths = [math.sqrt((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2) for s in segments]
    
    # We need to try all permutations of segments and all directions (start/end)
    # There are N! permutations and 2^N direction combinations
    # For N=6, 720 * 64 = 46,080 combinations, which is well within limits.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    # Generate all combinations of directions (0: A->C, 1: C->A)
    all_dirs = product([0, 1], repeat=N)
    
    # To avoid nested loops, we use a generator expression inside min()
    # We need to track the current position to calculate travel time S
    # Since we can't use loops, we'll use a helper function or a reduction-like 
    # approach to calculate the total time for a specific sequence.
    
    def calculate_time(perm, dirs):
        # Construct the sequence of points visited
        # Start at (0,0)
        # For each segment in perm:
        #   Move to start_point (speed S)
        #   Move to end_point (speed T)
        
        # Create the list of segments in the chosen order and direction
        ordered_segs = [
            (segments[i][0] if dirs[p] == 0 else segments[i][1],
             segments[i][1] if dirs[p] == 0 else segments[i][0])
            for p, i in zip(range(N), perm)
        ]
        
        # We need the travel time between segments.
        # Points: P0(0,0) -> Start1 -> End1 -> Start2 -> End2 ...
        # Travel times: 
        # dist(P0, Start1)/S + len1/T + dist(End1, Start2)/S + len2/T ...
        
        # Extract all points in the sequence
        points = [(0, 0)] + [pt for seg in ordered_segs for pt in seg]
        
        # Calculate travel times (non-emitting)
        # These are the gaps: (P0, S1), (E1, S2), (E2, S3)...
        # The gaps are at indices (0,1), (2,3), (4,5)...
        travel_times = [
            math.sqrt((points[i][0]-points[i+1][0])**2 + (points[i][1]-points[i+1][1])**2) / S
            for i in range(0, 2*N, 2)
        ]
        
        # Calculate printing times (emitting)
        # These are the segments: (S1, E1), (S2, E2)...
        # The segments are at indices (1,2), (3,4), (5,6)...
        print_times = [
            math.sqrt((points[i][0]-points[i+1][0])**2 + (points[i][1]-points[i+1][1])**2) / T
            for i in range(1, 2*N, 2)
        ]
        
        return sum(travel_times) + sum(print_times)

    # Since we cannot use loops, we use a comprehension to evaluate all possibilities
    # We use a nested comprehension to iterate over permutations and directions
    # Note: we must materialize the directions list because it's used inside the perm loop
    dirs_list = list(product([0, 1], repeat=N))
    
    ans = min(
        calculate_time(p, d) 
        for p in permutations(range(N)) 
        for d in dirs_list
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()