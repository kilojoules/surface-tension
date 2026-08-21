import sys
from itertools import permutations

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
        (
            (int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
            (int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        )
        for i in range(N)
    ]

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate the time to print each segment
    # print_times[i] is the time taken to move from one end to the other at speed T
    print_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments and all directions for each segment.
    # A state can be represented by (current_permutation, directions_bitmask).
    # However, since N is small (<= 6), we can use a generator expression inside 'min'.
    
    # For a given permutation of segments and a choice of directions:
    # directions: 0 means start at s[0] end at s[1], 1 means start at s[1] end at s[0].
    
    # We use a helper to calculate total time for a specific configuration.
    def calculate_total_time(perm, dirs):
        # Start at (0, 0)
        curr_pos = (0, 0)
        total_time = 0.0
        
        for i in range(N):
            seg_idx = perm[i]
            p1, p2 = segments[seg_idx]
            
            # Determine start and end points based on direction
            start_pt = p1 if dirs[i] == 0 else p2
            end_pt = p2 if dirs[i] == 0 else p1
            
            # Time to move to start point at speed S + time to print at speed T
            total_time += dist(curr_pos, start_pt) / S
            total_time += print_times[seg_idx]
            
            curr_pos = end_pt
            
        return total_time

    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # For each permutation, there are 2^N ways to orient the segments.
    # We can use a list comprehension to evaluate all 2^N directions for a given permutation.
    # To avoid explicit loops, we use a generator to feed into the min() function.
    
    # We use a trick to generate all binary strings of length N without a loop:
    # We can use a list comprehension to generate all tuples of 0s and 1s.
    all_dirs = [
        tuple((i >> j) & 1 for j in range(N)) 
        for i in range(1 << N)
    ]

    # The final answer is the minimum over all permutations and all direction combinations.
    ans = min(
        calculate_total_time(p, d)
        for p in all_perms
        for d in all_dirs
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()