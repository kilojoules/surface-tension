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
    
    segments = [
        (int(input_data[3 + 2*i]), int(input_data[4 + 2*i]), 
         int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        for i in range(N)
    ]

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate lengths of each segment
    seg_lengths = [dist((s[0], s[1]), (s[2], s[3])) for s in segments]
    
    # We need to try all permutations of segments and all 2^N orientations
    # A state is defined by (current_permutation, orientations)
    # Total complexity: N! * 2^N * N
    # For N=6, 720 * 64 * 6 is approx 276,480 operations, which fits in time.
    
    # Generate all permutations of indices 0...N-1
    perms = permutations(range(N))
    
    # For a fixed permutation, we can use DP to find the best orientations
    # dp[i][end_point_idx] = min time to finish first i segments, ending at endpoint idx (0 or 1)
    # However, since N is very small, we can just iterate through all 2^N orientations 
    # inside a comprehension or use a helper.
    
    # Let's define a function that calculates total time for a specific order and orientation
    def calc_time(p, orientations):
        # p: permutation of segment indices
        # orientations: tuple of 0 or 1 (0: start->end, 1: end->start)
        
        # Initial position
        curr_pos = (0, 0)
        total_time = 0.0
        
        for idx, ori in zip(p, orientations):
            seg = segments[idx]
            p_start = (seg[0], seg[1])
            p_end = (seg[2], seg[3])
            
            # Determine which point to move to and which to print to
            target_start = p_start if ori == 0 else p_end
            target_end = p_end if ori == 0 else p_start
            
            # Move to start (speed S), print to end (speed T)
            total_time += dist(curr_pos, target_start) / S
            total_time += seg_lengths[idx] / T
            curr_pos = target_end
            
        return total_time

    # We use a generator expression inside min() to avoid loops
    # We iterate over all permutations and all possible orientation combinations (2^N)
    # Using a list comprehension to generate all orientation tuples
    all_oris = [tuple((i >> j) & 1 for j in range(N)) for i in range(1 << N)]
    
    # The result is the minimum time across all permutations and orientations
    # Note: the orientation tuple corresponds to the segments in the given permutation
    # So we map the orientation bits to the segments in the permutation.
    
    ans = min(
        calc_time(p, ori) 
        for p in perms 
        for ori in all_oris
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()