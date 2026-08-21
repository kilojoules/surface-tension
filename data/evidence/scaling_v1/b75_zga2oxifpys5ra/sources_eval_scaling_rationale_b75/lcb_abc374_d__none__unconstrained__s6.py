import sys
import math
from itertools import permutations, product

def solve():
    # Read input and parse values
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples ((x1, y1), (x2, y2))
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    # Helper to calculate Euclidean distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Generate all permutations of segment indices
    # Generate all possible orientations (0: start->end, 1: end->start) for each segment
    # We use a list comprehension to calculate the total time for every possible combination
    # The state is tracked by mapping the sequence of points visited.
    
    # For a given permutation of segments and a given orientation:
    # Points visited: P0(0,0) -> Start1 -> End1 -> Start2 -> End2 ...
    # Time = dist(P0, Start1)/S + dist(Start1, End1)/T + dist(End1, Start2)/S + ...
    
    all_results = [
        sum(
            # Time to move to the start of the segment (except for the first segment's start from 0,0)
            # plus time to print the segment.
            # We construct the sequence of points: (0,0), (s1_start, s1_end), (s2_start, s2_end)...
            # The distance between End_{i-1} and Start_i is divided by S.
            # The distance between Start_i and End_i is divided by T.
            [
                dist(
                    (0, 0) if i == 0 else (
                        p[i-1][1][0], p[i-1][1][1]
                    ),
                    (p[i][0][0], p[i][0][1])
                ) / S,
                dist(p[i][0], p[i][1]) / T
            ]
            for i in range(N)
        )
        # We flatten the nested list created by the comprehension inside sum()
        # Actually, the structure above returns a list of lists, so we sum the flattened version.
        for p in [
            # For each permutation of segments, try all 2^N orientations
            # p is a sequence of (start_point, end_point)
            # We use a nested comprehension to handle the sum of the pairs
            # But since we can't use loops, we map the logic into a single expression.
            # Let's redefine the logic slightly to fit a single sum() call.
            None 
        ]
    ]
    
    # Corrected approach using a generator expression inside min()
    # 1. Permute the segments
    # 2. For each permutation, product(range(2), repeat=N) defines the direction
    # 3. Calculate total time
    
    ans = min(
        sum(
            # For each segment in the chosen order and direction:
            # Move time from previous end point to current start point + Print time
            (
                dist(
                    (0, 0) if i == 0 else (
                        # Previous segment's end point
                        # perm[i-1][1] if direction[i-1] == 0 else perm[i-1][0]
                        # Wait, the logic is simpler: 
                        # Let current_seg = segments[perm[i]]
                        # If dir[i] == 0: start=A, end=B. Else: start=B, end=A.
                        # prev_end is the end point of the previous segment.
                        # We can pre-calculate the oriented segments.
                        oriented[i-1][1]
                    ),
                    oriented[i][0]
                ) / S + dist(oriented[i][0], oriented[i][1]) / T
            )
            for i in range(N)
        )
        for perm in permutations(range(N))
        for dirs in product([0, 1], repeat=N)
        for oriented in [
            # Create the oriented segments based on the permutation and directions
            [
                (segments[perm[i]][0], segments[perm[i]][1]) if dirs[i] == 0 
                else (segments[perm[i]][1], segments[perm[i]][0])
                for i in range(N)
            ]
        ]
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()