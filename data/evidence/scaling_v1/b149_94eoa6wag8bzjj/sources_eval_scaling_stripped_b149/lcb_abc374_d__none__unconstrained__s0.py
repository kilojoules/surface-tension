import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input
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

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Precompute the time to print each segment
    # print_times[i] is the time taken to move from one end to the other at speed T
    print_times = [dist(s[0], s[1]) / T for s in segments]
    
    # We need to try all permutations of segments and all possible directions for each segment.
    # A direction is represented by whether we end at endpoint 0 or endpoint 1.
    # There are N! permutations and 2^N direction combinations.
    # Since N <= 6, N! * 2^N is at most 720 * 64 = 46,080, which is well within limits.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    
    # For a fixed permutation, we can use dynamic programming or a recursive approach 
    # to find the best directions. However, with N=6, we can just iterate all 2^N.
    # But wait, we can't use 'for' loops. We can use map/reduce.
    
    # Let's define a function that calculates the total time for a specific 
    # permutation of segments and a specific choice of directions.
    # directions: a tuple of 0 or 1. If dir[i] == 0, we print from s[1] to s[0].
    # If dir[i] == 1, we print from s[0] to s[1].
    
    def calc_time(perm, dirs):
        # Current position starts at (0, 0)
        # We need to calculate:
        # Move to start of seg 1 + print seg 1 + move to start of seg 2 + print seg 2 ...
        
        # Extract the endpoints based on the chosen directions
        # endpoints[i] = (start_point, end_point)
        endpoints = [
            (segments[perm[i]][1], segments[perm[i]][0]) if dirs[i] == 0 
            else (segments[perm[i]][0], segments[perm[i]][1])
            for i in range(N)
        ]
        
        # Calculate travel times between segments
        # Travel 0: (0,0) to endpoints[0][0]
        # Travel i: endpoints[i-1][1] to endpoints[i][0]
        travel_times = [
            dist((0, 0), endpoints[0][0]) / S
        ] + [
            dist(endpoints[i-1][1], endpoints[i][0]) / S
            for i in range(1, N)
        ]
        
        return sum(travel_times) + sum(print_times)

    # Generate all possible direction combinations (0 or 1 for each segment)
    # Using a list comprehension to generate 2^N combinations
    all_dirs = [
        tuple((permutation_of_bits >> i) & 1 for i in range(N))
        for permutation_of_bits in range(1 << N)
    ]

    # Use map and reduce to find the minimum time across all permutations and directions
    # We map each permutation to the minimum time found among all direction combinations
    ans = min(
        map(
            lambda perm: min(map(lambda dirs: calc_time(perm, dirs), all_dirs)),
            all_perms
        )
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()