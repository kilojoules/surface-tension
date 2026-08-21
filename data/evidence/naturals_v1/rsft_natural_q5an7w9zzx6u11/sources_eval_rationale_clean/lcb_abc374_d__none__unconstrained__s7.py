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

    # Precompute printing time for each segment
    print_times = [dist(s[0], s[1]) / T for s in segments]
    
    # We need to try all permutations of segments and all directions (2^N)
    # A direction 0 means start at s[0] end at s[1], 1 means start at s[1] end at s[0]
    
    # Generate all permutations of indices 0...N-1
    all_perms = permutations(range(N))
    # Generate all possible direction combinations (0 or 1 for each segment)
    all_dirs = list(product([0, 1], repeat=N))
    
    def calculate_total_time(perm, dirs):
        # Determine the sequence of points visited
        # Start at (0,0)
        points = [(0, 0)]
        
        # For each segment in the permutation, determine start and end points based on dirs
        # We use a list comprehension to build the sequence of endpoints
        path = [
            (segments[i][dirs[i]], segments[i][1 - dirs[i]])
            for i in perm
        ]
        
        # The total time is the sum of:
        # 1. Time to print all segments (constant regardless of order/direction)
        # 2. Time to move between segments (depends on order/direction)
        
        # To calculate move time, we need the end of segment k and start of segment k+1
        # move_segments = [(path[i][1], path[i+1][0]) for i in range(N-1)]
        # But we can't use for loops. We use zip.
        
        # Start point for the first move is (0,0), then the end of the first segment
        starts = [ (0,0) ] + [p[1] for p in path]
        ends = [p[0] for p in path]
        
        move_dist = sum(map(lambda pair: dist(pair[0], pair[1]), zip(starts, ends)))
        
        return (move_dist / S) + sum(print_times)

    # We need to find the minimum over all permutations and all direction combinations.
    # Since N is small (<= 6), N! * 2^N is at most 720 * 64 = 46,080.
    
    # Use a generator expression to find the minimum
    ans = min(
        calculate_total_time(p, d)
        for p in all_perms
        for d in all_dirs
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()