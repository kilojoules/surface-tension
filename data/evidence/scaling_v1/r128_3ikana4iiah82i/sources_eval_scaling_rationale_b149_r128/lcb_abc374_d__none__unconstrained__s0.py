import sys
import math
from itertools import permutations, product

def solve():
    # Read input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into pairs of endpoints
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]
    
    # Helper to calculate Euclidean distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # We need to try all permutations of the N segments
    # For each segment, we can start at either endpoint A or B
    # There are N! * 2^N possible paths
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # Generate all possible direction configurations (0 for A->B, 1 for B->A)
    all_dirs = product([0, 1], repeat=N)
    
    # To avoid nested loops, we use a generator expression.
    # We iterate over all permutations and all direction configurations.
    # For a fixed permutation and direction set, we calculate the total time.
    
    # Since we cannot use loops, we must structure the calculation of a single path's time
    # as a reduction or a sum of segments.
    
    def calculate_path_time(perm, dirs):
        # Determine the sequence of points visited
        # Start at (0,0)
        # For each segment in perm:
        #   Start point is segments[perm[i]][dirs[i]]
        #   End point is segments[perm[i]][1 - dirs[i]]
        
        # Create the sequence of (start, end) pairs for the chosen path
        path = [
            (segments[perm[i]][dirs[i]], segments[perm[i]][1 - dirs[i]])
            for i in range(N)
        ]
        
        # The total time is:
        # Time to move from (0,0) to path[0].start + 
        # Sum of (time to print segment + time to move to next segment's start)
        
        # Time to print all segments is constant regardless of order/direction
        print_time = sum(dist(s[0], s[1]) for s in segments) / T
        
        # Time to move between segments (and from origin)
        # We need the distance from (0,0) to path[0].start
        # and from path[i].end to path[i+1].start
        
        # Using a list comprehension to get all transition distances
        transitions = [
            dist((0, 0), path[0][0])
        ] + [
            dist(path[i][1], path[i+1][0])
            for i in range(N - 1)
        ]
        
        move_time = sum(transitions) / S
        
        return print_time + move_time

    # We use a generator expression inside min() to find the optimal time.
    # Note: all_dirs is a product, so we need to handle it carefully.
    # Because we can't use loops, we map the calculation over the Cartesian product.
    
    # We need to evaluate all combinations of permutations and directions.
    # Since N is small (<=6), N! * 2^N is at most 720 * 64 = 46,080.
    
    # We use a nested generator: for each permutation, try all direction combinations.
    ans = min(
        calculate_path_time(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()