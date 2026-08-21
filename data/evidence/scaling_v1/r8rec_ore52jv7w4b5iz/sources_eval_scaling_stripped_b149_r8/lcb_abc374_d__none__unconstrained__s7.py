import sys
from itertools import permutations, product
from functools import reduce

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

    # Precompute segment lengths and printing times
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    print_times = [l / T for l in seg_lengths]

    # We need to try all permutations of segments and all possible directions for each segment.
    # A direction is represented by 0 (start -> end) or 1 (end -> start).
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    # Generate all possible direction combinations (2^N)
    directions = product([0, 1], repeat=N)
    
    # To avoid loops, we can't use a generator for directions inside the perm loop 
    # if we want to use min() over a comprehension. 
    # Since N is small (<= 6), we can pre-calculate all direction tuples.
    all_dirs = list(directions)

    # For a fixed permutation of segments and a fixed set of directions:
    # The total time is:
    # Time to move to 1st segment start + print 1st + move to 2nd start + print 2nd ...
    
    # We use a helper to calculate the total travel time for a specific configuration.
    def calculate_time(perm, dir_config):
        # Current position starts at (0, 0)
        # We need to track (current_pos, total_travel_time)
        # The sequence of points visited is:
        # Start -> Seg[p0]_start -> Seg[p0]_end -> Seg[p1]_start -> Seg[p1]_end ...
        
        # Map permutation and directions to the actual sequence of points
        points = [ (0, 0) ]
        for i in range(N):
            seg_idx = perm[i]
            d = dir_config[i]
            p_start = segments[seg_idx][0] if d == 0 else segments[seg_idx][1]
            p_end = segments[seg_idx][1] if d == 0 else segments[seg_idx][0]
            points.append(p_start)
            points.append(p_end)
            
        # Travel time (without laser) is distance between p_end_i and p_start_{i+1}
        # The distance from (0,0) to the first p_start is also travel time.
        # The distance from p_start_i to p_end_i is printing time.
        
        # Travel distances: (points[0], points[1]), (points[2], points[3]), (points[4], points[5])...
        # These are the "jumps" between segments.
        travel_dist = sum(dist(points[2*i], points[2*i+1]) for i in range(N))
        
        # Printing distances: (points[1], points[2]), (points[3], points[4])...
        # Actually, the printing time is constant regardless of order, 
        # but the travel time depends on the sequence.
        return (travel_dist / S) + sum(print_times)

    # We need to find the minimum time across all permutations and direction configurations.
    # Since we cannot use loops, we use a nested comprehension.
    # Note: the dir_config needs to be mapped to the permutation.
    # If dir_config is (d0, d1, ... dN), it means the i-th segment in the permutation is printed in direction di.
    
    ans = min(
        calculate_time(p, d)
        for p in perms
        for d in all_dirs
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()