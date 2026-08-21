import sys
from itertools import permutations
import math

def solve():
    # Read input and parse N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples: ((x1, y1), (x2, y2))
    segments = [
        (
            (int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
            (int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        )
        for i in range(N)
    ]
    
    # Precompute lengths of each segment
    # Time to print segment i is length / T
    seg_times = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2) / T
        for s in segments
    ]
    
    # We need to visit every segment. For each segment, we can start at either end.
    # There are N! permutations of segments and 2^N ways to choose directions.
    # Since N is small (<= 6), we can iterate through all permutations and 
    # use dynamic programming or recursion to find the best direction for each.
    
    # For a fixed order of segments, the state is (current_segment_index, end_point_used).
    # However, with N=6, we can simply use itertools.product to try all 2^N directions
    # for every permutation.
    
    from itertools import product
    
    # Function to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    # Generate all possible direction choices (0: start->end, 1: end->start)
    dirs = product([0, 1], repeat=N)
    
    # To optimize, we can group the logic. For a fixed permutation, 
    # we want to minimize the travel time between segments.
    # Let f(i, side) be the min time to finish i segments ending at side of segment i.
    # But since N is only 6, we can use a comprehension over permutations and 
    # a nested comprehension/generator over the 2^N direction combinations.
    
    # We use a helper to calculate total time for a specific order and direction set.
    def calc_time(p, d):
        # p: permutation of indices, d: tuple of directions (0 or 1)
        # Current position starts at (0, 0)
        # For each segment i in p:
        #   Start point is segments[i][d[i]], End point is segments[i][1-d[i]]
        #   Travel time = dist(current, start) / S + length(segment) / T
        
        # We can use a list comprehension with a custom accumulator via a loop 
        # or a reduction, but a simple generator expression inside sum() 
        # requires tracking the 'current' position. 
        # Since we cannot use loops, we use a recursive-like approach or 
        # map/reduce. Actually, the most 'Pythonic' way without explicit for-loops
        # to handle state is using a reduction.
        
        from functools import reduce
        
        def step(state, idx_dir):
            idx, direction = idx_dir
            curr_pos, total_t = state
            p1, p2 = segments[idx]
            start = p1 if direction == 0 else p2
            end = p2 if direction == 0 else p1
            
            travel_t = dist(curr_pos, start) / S
            print_t = seg_times[idx]
            return (end, total_t + travel_t + print_t)
        
        # Initial state: ((0,0), 0)
        # Sequence of (index, direction)
        seq = ((p[i], d[i]) for i in range(N))
        final_state = reduce(step, seq, ((0, 0), 0.0))
        return final_state[1]

    # We need to iterate over all permutations and all direction combinations.
    # To avoid explicit loops, we use a generator expression inside min().
    # Note: dirs is a generator, so we convert it to a list to reuse it for each permutation.
    dir_options = list(product([0, 1], repeat=N))
    
    # For each permutation p, we find the best direction combination d.
    # Then we find the minimum across all permutations.
    ans = min(
        min(calc_time(p, d) for d in dir_options)
        for p in perms
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()