import sys
import math
from itertools import permutations, product
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments as tuples of (x1, y1, x2, y2)
    segments = [
        (int(input_data[3 + 4*i]), int(input_data[4 + 4*i]), 
         int(input_data[5 + 4*i]), int(input_data[6 + 4*i]))
        for i in range(N)
    ]

    # Precompute length of each segment
    lengths = [
        math.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2) 
        for s in segments
    ]

    # Function to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # For a given permutation of segments and a given set of directions:
    # directions: 0 means (A, B) -> (C, D), 1 means (C, D) -> (A, B)
    calculate_time = lambda perm, dirs: (
        reduce(
            lambda acc, item: (
                # acc: (current_x, current_y, total_time)
                # item: (segment_index, direction)
                (
                    # New position is the end point of the segment
                    (
                        segments[item[0]][2] if item[1] == 0 else segments[item[0]][0],
                        segments[item[0]][3] if item[1] == 0 else segments[item[0]][1]
                    ),
                    # New time: old time + move to start + print segment
                    acc[2] + 
                    dist(
                        (acc[0], acc[1]), 
                        (segments[item[0]][0] if item[1] == 0 else segments[item[0]][2],
                         segments[item[0]][1] if item[1] == 0 else segments[item[0]][3])
                    ) / S + 
                    lengths[item[0]] / T
                )
            ),
            # Zip the permutation of indices with the chosen directions
            zip(perm, dirs),
            # Initial state: laser at (0, 0), time 0
            (0, 0, 0.0)
        )[2]
    )

    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # Generate all possible direction combinations (2^N)
    all_dirs = list(product([0, 1], repeat=N))

    # Find the minimum time across all permutations and direction combinations
    # We use a nested list comprehension to evaluate all possibilities
    result = min(
        calculate_time(p, d) 
        for p in all_perms 
        for d in all_dirs
    )

    print(f"{result:.20f}")

if __name__ == "__main__":
    solve()