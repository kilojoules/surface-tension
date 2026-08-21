import sys
from itertools import permutations

def solve():
    # Read input and parse N, S, T
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
    # print_times[i] = length of segment i / T
    print_times = [dist(s[0], s[1]) / T for s in segments]
    
    # We need to try all permutations of segments and all directions for each segment.
    # A state can be represented by (current_permutation, directions_bitmask).
    # However, since N is small (<= 6), we can use a generator expression inside min().
    
    # For a given permutation of segments and a choice of directions:
    # directions is a tuple of 0 or 1. 
    # If dir=0, segment is printed from p0 to p1. If dir=1, from p1 to p0.
    
    # We use a helper to calculate total time for a specific configuration.
    def calc_total_time(perm, dirs):
        # Starting position
        curr_pos = (0, 0)
        total_time = 0.0
        
        for i in range(N):
            seg_idx = perm[i]
            p0, p1 = segments[seg_idx]
            
            # Determine start and end of the print stroke based on direction
            start_p = p0 if dirs[i] == 0 else p1
            end_p = p1 if dirs[i] == 0 else p0
            
            # Time to move to start point (at speed S) + time to print (at speed T)
            total_time += dist(curr_pos, start_p) / S
            total_time += print_times[seg_idx]
            
            curr_pos = end_p
            
        return total_time

    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # Generate all possible direction combinations (2^N)
    # We can use a list comprehension to generate all binary strings of length N
    all_dirs = [
        tuple((perm_dirs >> i) & 1 for i in range(N))
        for perm_dirs in range(1 << N)
    ]
    
    # Find the minimum time across all permutations and direction combinations
    # We use a nested generator to avoid creating large lists in memory
    ans = min(
        calc_total_time(p, d)
        for p in all_perms
        for d in all_dirs
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()