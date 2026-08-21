import sys
from itertools import permutations

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples: ((ax, ay), (cx, cy))
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate the time to print each segment
    # print_times[i] is the time taken to move from one end to the other at speed T
    print_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments and all possible directions for each segment.
    # A direction is represented by whether we start at endpoint 0 or endpoint 1.
    # There are N! permutations and 2^N direction combinations.
    # Since N <= 6, N! * 2^N is at most 720 * 64 = 46,080, which is well within limits.
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    
    # For a fixed permutation, we can use dynamic programming to find the best directions.
    # dp[i][end] = minimum time to print first i segments, ending at endpoint 'end' of segment i.
    # However, with N=6, we can just use a comprehension to simulate the 2^N choices 
    # or use a recursive approach with memoization.
    
    def get_min_time(perm):
        # state: (current_position, current_time)
        # We start at (0, 0)
        # For the first segment in the permutation:
        s0 = segments[perm[0]]
        # Option 1: Start at s0[0], end at s0[1]
        # Option 2: Start at s0[1], end at s0[0]
        
        # We use a list of (end_point, total_time) for the current step
        # Initial state for the first segment
        states = [
            (s0[1], dist((0,0), s0[0])/S + print_times[perm[0]]),
            (s0[0], dist((0,0), s0[1])/S + print_times[perm[0]])
        ]
        
        for i in range(1, N):
            seg_idx = perm[i]
            s = segments[seg_idx]
            p_time = print_times[seg_idx]
            
            # For each current state, try both directions of the next segment
            # New states will be (end_point, time)
            # We only need to keep the minimum time for each of the two possible end points
            
            # Try ending at s[1] (starting at s[0])
            time_end_s1 = min(st[1] + dist(st[0], s[0])/S + p_time for st in states)
            # Try ending at s[0] (starting at s[1])
            time_end_s0 = min(st[1] + dist(st[0], s[1])/S + p_time for st in states)
            
            states = [(s[1], time_end_s1), (s[0], time_end_s0)]
            
        return min(st[1] for st in states)

    # Calculate the minimum time across all permutations
    ans = min(get_min_time(p) for p in perms)
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()