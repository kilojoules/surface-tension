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

    # Precompute lengths of segments for printing time
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    total_print_time = sum(seg_lengths) / T

    # We need to find the minimum travel time between segments.
    # There are N! permutations of segments and 2^N combinations of directions.
    # A direction choice of 0 means start at s[0] end at s[1], 1 means vice versa.
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    # Generate all combinations of directions (0 or 1)
    all_dirs = product([0, 1], repeat=N)

    # To avoid nested loops, we use a generator expression inside min()
    # We calculate the travel time for each permutation and direction set.
    # Travel time = dist(start, first_start)/S + sum(dist(prev_end, next_start)/S)
    
    # Since we can't use loops, we'll use a helper function to calculate 
    # the travel cost for a specific sequence and direction set.
    def get_travel_cost(perm, dirs):
        # Create the sequence of points: (start1, end1, start2, end2, ...)
        points = [
            (segments[perm[i]][dirs[i]], segments[perm[i]][1 - dirs[i]])
            for i in range(N)
        ]
        
        # The sequence of movements is:
        # (0,0) -> points[0][0]
        # points[0][1] -> points[1][0]
        # points[1][1] -> points[2][0] ...
        
        # Starting move
        start_move = dist((0, 0), points[0][0])
        
        # Intermediate moves
        # Use a list comprehension to get distances between segments
        inter_moves = [
            dist(points[i][1], points[i+1][0])
            for i in range(N - 1)
        ]
        
        return (start_move + sum(inter_moves)) / S

    # We need to iterate over all permutations and all direction combinations.
    # Because we cannot use nested loops, we use a generator expression.
    # Note: N is small (up to 6), so N! * 2^N is at most 720 * 64 = 46,080.
    
    # We use a nested generator to evaluate all combinations.
    # We use a list for dirs inside the permutation loop to avoid re-generating 2^N every time.
    dir_options = list(product([0, 1], repeat=N))
    
    min_travel_time = min(
        get_travel_cost(p, d)
        for p in permutations(range(N))
        for d in dir_options
    )

    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()