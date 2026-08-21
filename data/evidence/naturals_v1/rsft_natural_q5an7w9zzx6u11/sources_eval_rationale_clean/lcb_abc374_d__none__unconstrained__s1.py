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

    # We need to try all permutations of segments and all directions (start/end)
    # There are N! permutations and 2^N direction combinations.
    # N <= 6, so 720 * 64 = 46,080 combinations, which is feasible.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    # Generate all possible direction flips (0: A->C, 1: C->A)
    all_dirs = product([0, 1], repeat=N)

    # To avoid loops, we use a generator expression to calculate total time for each configuration.
    # For a fixed permutation and direction set:
    # Total Time = Sum(Printing Times) + Sum(Travel Times)
    # Printing Time is constant regardless of order: sum(lengths) / T
    
    total_print_time = sum(seg_lengths) / T

    def calculate_travel_time(perm, dirs):
        # We need to calculate the distance from (0,0) to 1st start, 
        # then 1st end to 2nd start, etc.
        # Since we can't use loops, we'll create a list of points visited.
        
        # Get the points in order based on permutation and direction
        # points[i] = (start_i, end_i)
        ordered_segs = [
            (segments[perm[i]][dirs[perm[i]]], segments[perm[i]][1 - dirs[perm[i]]])
            for i in range(N)
        ]
        
        # The sequence of movements is:
        # (0,0) -> start_0 -> end_0 -> start_1 -> end_1 ...
        # Travel distances are:
        # dist((0,0), start_0) + dist(end_0, start_1) + ... + dist(end_{N-2}, start_{N-1})
        
        # Use a list comprehension to get the travel segments
        travel_pairs = [
            ((0, 0), ordered_segs[0][0])
        ] + [
            (ordered_segs[i][1], ordered_segs[i+1][0])
            for i in range(N - 1)
        ]
        
        return sum(dist(p1, p2) for p1, p2 in travel_pairs) / S

    # We use a nested generator to find the minimum travel time.
    # Note: we must wrap all_dirs in a list or recreate it because it's an iterator.
    # However, since we need to iterate it for every permutation, 
    # it's easier to just use a list comprehension inside the min().
    
    # To strictly avoid loops, we use map/sum/min and generator expressions.
    # We pre-calculate the direction combinations as a list.
    dir_combos = list(product([0, 1], repeat=N))
    
    ans = total_print_time + min(
        sum(
            dist(
                (0, 0) if i == 0 else segments[p[i-1]][1 - dirs[p[i-1]]],
                segments[p[i]][dirs[p[i]]]
            )
            for i in range(N)
        ) / S
        for p in permutations(range(N))
        for dirs in dir_combos
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()