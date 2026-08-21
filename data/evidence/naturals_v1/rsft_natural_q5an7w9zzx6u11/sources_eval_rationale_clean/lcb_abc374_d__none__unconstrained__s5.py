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
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute segment lengths and printing times
    seg_times = [dist(s[0], s[1]) / T for s in segments]
    
    # We need to try all permutations of segments and all directions for each segment
    # There are N! permutations and 2^N direction combinations
    # N <= 6, so 720 * 64 = 46,080 combinations, which is well within limits.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    
    # Generate all possible direction choices (0: start->end, 1: end->start)
    all_dirs = list(product([0, 1], repeat=N))
    
    # Function to calculate total time for a specific permutation and direction set
    def calc_time(perm, dirs):
        # Current position starts at (0, 0)
        # We need to track the sequence of points visited.
        # For each segment in perm, we determine the start and end points based on dirs.
        
        # Create the sequence of (start_point, end_point) for the chosen perm and dirs
        path = [
            (segments[perm[i]][0] if dirs[perm[i]] == 0 else segments[perm[i]][1],
             segments[perm[i]][1] if dirs[perm[i]] == 0 else segments[perm[i]][0])
            for i in range(N)
        ]
        
        # The total time is the sum of:
        # 1. Time to move from (0,0) to the first start point
        # 2. Time to print each segment
        # 3. Time to move from the end of segment i to the start of segment i+1
        
        # Starting point
        p_start = (0, 0)
        
        # Move to first segment
        move_0 = dist(p_start, path[0][0]) / S
        
        # Printing times (constant regardless of order/direction)
        print_total = sum(seg_times)
        
        # Intermediate moves
        # Use a list comprehension to calculate distances between segments
        moves_between = [
            dist(path[i][1], path[i+1][0]) / S 
            for i in range(N - 1)
        ]
        
        return move_0 + print_total + sum(moves_between)

    # Evaluate all combinations and find the minimum
    # Using a nested comprehension to iterate through permutations and directions
    ans = min(
        calc_time(p, d) 
        for p in all_perms 
        for d in all_dirs
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()