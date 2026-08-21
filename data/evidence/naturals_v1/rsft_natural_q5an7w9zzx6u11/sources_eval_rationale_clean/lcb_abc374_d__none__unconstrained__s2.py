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

    # Precompute lengths of segments for the T-speed part
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    total_print_time = sum(seg_lengths) / T

    # We need to find the minimum travel time (S-speed part)
    # There are N! permutations of segments and 2^N ways to orient them.
    # A state is defined by the current position.
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    # Generate all possible orientations (0: start->end, 1: end->start)
    all_orientations = product([0, 1], repeat=N)

    def calculate_travel_dist(perm, orient):
        # Create the sequence of points visited
        # Start at (0,0)
        points = [(0, 0)]
        
        # For each segment in the permutation, determine the start and end points
        # based on the orientation
        path_points = [
            (segments[perm[i]][orient[i]], segments[perm[i]][1 - orient[i]])
            for i in range(N)
        ]
        
        # The travel distance is the distance from current point to the start of the next segment.
        # We use a generator to calculate distances between the end of segment i and start of segment i+1.
        # The first travel is from (0,0) to the start of the first segment.
        
        # To avoid loops, we construct a list of all "jump" pairs.
        # Jump 0: (0,0) -> path_points[0][0]
        # Jump i: path_points[i-1][1] -> path_points[i][0]
        
        starts = [p[0] for p in path_points]
        ends = [p[1] for p in path_points]
        
        # Travel segments: (0,0)->starts[0], ends[0]->starts[1], ..., ends[N-2]->starts[N-1]
        travel_pairs = [((0, 0), starts[0])] + [
            (ends[i], starts[i+1]) for i in range(N-1)
        ]
        
        return sum(dist(p1, p2) for p1, p2 in travel_pairs)

    # We need to iterate over all permutations and orientations.
    # Since we can't use nested loops, we use a generator expression.
    # Note: we must realize that all_orientations needs to be recreated for each permutation
    # or we can product them together.
    
    # To avoid nested loops and recursion, we use a flat generator.
    # We use a helper to get the minimum travel distance.
    
    # Because N is small (<= 6), N! * 2^N is at most 720 * 64 = 46,080.
    # We can use a nested generator expression.
    
    min_travel_dist = min(
        sum(
            dist(
                (0, 0) if i == 0 else segments[perm[i-1]][1 - orients[i-1]],
                segments[perm[i]][orients[i]]
            )
            for i in range(N)
        )
        for perm in permutations(range(N))
        for orients in product([0, 1], repeat=N)
    )

    print(f"{total_print_time + min_travel_dist / S:.20f}")

if __name__ == "__main__":
    solve()