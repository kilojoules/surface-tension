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

    # Precompute lengths of segments for T-speed printing
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    
    # We need to try all permutations of segments and all directions (start/end)
    # There are N! permutations and 2^N direction combinations
    # N <= 6, so 720 * 64 = 46,080 combinations, which is well within limits.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    # Generate all possible direction flips (0: A->C, 1: C->A)
    all_dirs = product([0, 1], repeat=N)

    # To avoid loops, we use a generator expression inside min()
    # For a fixed permutation and direction set:
    # Total time = sum(dist(current_pos, next_start)/S) + sum(seg_length/T)
    
    # The sum of seg_length/T is constant regardless of order
    constant_print_time = sum(l / T for l in seg_lengths)
    
    # We need to find the minimum travel time between segments
    # travel_time = dist(0,0 to start1)/S + dist(end1 to start2)/S + ...
    
    # Since we cannot use loops, we use a helper function to calculate travel time for a sequence
    def calc_travel_time(perm, dirs):
        # Create the sequence of points: (start1, end1, start2, end2, ...)
        points = [
            (segments[i][dirs[i]], segments[i][1 - dirs[i]])
            for i in perm
        ]
        
        # Flatten the points into a list of coordinates
        # We use a list comprehension here as it is allowed
        coords = [p for seg in points for p in seg]
        
        # Calculate distances between (0,0)->p0, p1->p2, p3->p4...
        # The travel segments are at indices (0, 2, 4...) relative to the flattened list
        # But it's easier to think: 
        # Travel 0: (0,0) to coords[0]
        # Travel 1: coords[1] to coords[2]
        # Travel 2: coords[3] to coords[4]...
        
        # We can use a list comprehension to get the pairs and sum their distances
        travel_dists = [
            dist((0, 0), coords[0])
        ] + [
            dist(coords[i], coords[i+1]) 
            for i in range(1, 2*N, 2)
        ]
        
        return sum(travel_dists) / S

    # To strictly avoid 'for' loops in the logic, we use map/itertools
    # However, the prompt allows list comprehensions. 
    # Let's refine the travel time calculation to be a one-liner.
    
    # We iterate over all permutations and all direction combinations
    # We use a generator expression to find the minimum travel time.
    
    # Since we need to iterate over all_dirs for every perm, 
    # and all_dirs is a product object (exhaustible), we must recreate it or use a list.
    dirs_list = list(product([0, 1], repeat=N))
    
    min_travel = min(
        sum(
            [
                dist((0, 0), segments[perm[0]][d[perm[0]]]) / S
            ] + [
                dist(segments[perm[i-1]][1 - d[perm[i-1]]], segments[perm[i]][d[perm[i]]]) / S
                for i in range(1, N)
            ]
        )
        for perm in permutations(range(N))
        for d in dirs_list
    )

    print(f"{min_travel + constant_print_time:.20f}")

if __name__ == "__main__":
    solve()