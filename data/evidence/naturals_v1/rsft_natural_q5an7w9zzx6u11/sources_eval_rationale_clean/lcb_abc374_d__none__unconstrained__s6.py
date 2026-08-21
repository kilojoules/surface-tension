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
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    print_times = [l / T for l in seg_lengths]

    # We need to try all permutations of segments and all directions (start/end)
    # There are N! permutations and 2^N direction combinations
    # For N=6, 720 * 64 = 46,080 combinations, which is well within limits.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    # Generate all combinations of directions (0: A->C, 1: C->A)
    all_dirs = product([0, 1], repeat=N)

    # To avoid nested loops, we use a generator expression inside min()
    # We calculate the total time for a specific permutation and direction set
    # Total Time = Sum(printing times) + Sum(travel times)
    
    # The sum of printing times is constant regardless of order
    total_print_time = sum(print_times)

    def calculate_travel_time(perm, dirs):
        # Determine the sequence of points visited
        # Start at (0,0), then for each segment in perm:
        # move to start_point, then print to end_point
        
        # Construct the sequence of points
        points = [(0, 0)]
        # We use a list comprehension to build the sequence of endpoints
        # For each segment i in perm, if dirs[i]==0: (A, B) -> (C, D), else (C, D) -> (A, B)
        
        # Since we cannot use loops, we build the sequence of points 
        # by mapping the permutation and directions
        seq = [
            (segments[p][dirs[p]], segments[p][1 - dirs[p]]) 
            for p in perm
        ]
        
        # The travel distance is:
        # dist((0,0), seq[0].start) + dist(seq[0].end, seq[1].start) + ...
        
        # Start point of first segment
        d0 = dist((0, 0), seq[0][0])
        
        # Distances between segments: end of i to start of i+1
        # Use a list comprehension to calculate distances between consecutive segments
        inter_dists = [
            dist(seq[i][1], seq[i+1][0]) 
            for i in range(N - 1)
        ]
        
        return (d0 + sum(inter_dists)) / S

    # We need to iterate over all permutations and all direction combinations.
    # To avoid loops, we use a nested generator expression.
    # Note: all_dirs is a product, so we convert it to a list to reuse it or 
    # wrap it in the comprehension.
    
    # Because we need to iterate all_dirs for every permutation, 
    # and all_dirs is an iterator, we convert it to a tuple.
    dirs_tuple = tuple(product([0, 1], repeat=N))
    
    ans = min(
        total_print_time + calculate_travel_time(p, d)
        for p in permutations(range(N))
        for d in dirs_tuple
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()