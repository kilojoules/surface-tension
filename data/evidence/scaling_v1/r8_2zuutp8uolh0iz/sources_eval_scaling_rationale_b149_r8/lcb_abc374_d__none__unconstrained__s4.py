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
        segments.append((int(input_data[idx]), int(input_data[idx+1]), 
                         int(input_data[idx+2]), int(input_data[idx+3])))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute lengths of segments
    seg_lengths = [dist((s[0], s[1]), (s[2], s[3])) for s in segments]
    
    # We need to try all permutations of segments and all directions (start/end)
    # There are N! permutations and 2^N direction combinations
    # Since N <= 6, N! * 2^N is at most 720 * 64 = 46080, which is small.
    
    # Generate all permutations of indices
    perms = permutations(range(N))
    
    # For a fixed permutation, we can use DP or brute force the directions.
    # Let's use a list comprehension to evaluate all direction combinations for each permutation.
    
    def evaluate_path(perm, directions):
        # directions is a tuple of 0 or 1 (0: A->C, 1: C->A)
        current_pos = (0, 0)
        total_time = 0.0
        
        # We use a helper to simulate the path. Since we can't use loops, 
        # we use a trick with a list and a reducing-like structure.
        # However, the state (current_pos, total_time) must be passed along.
        # We can use a custom reduce-like approach via a list comprehension 
        # and a mutable object or just map the logic.
        
        # To avoid loops/recursion, we pre-calculate the endpoints based on directions.
        points = [
            (
                (segments[i][0], segments[i][1]) if directions[j] == 0 else (segments[i][2], segments[i][3]),
                (segments[i][2], segments[i][3]) if directions[j] == 0 else (segments[i][0], segments[i][1])
            )
            for j, i in enumerate(perm)
        ]
        
        # Calculate travel times between segments
        # Start to first segment
        start_dist = dist(current_pos, points[0][0])
        
        # Times to print segments
        print_times = [seg_lengths[i] / T for i in perm]
        
        # Times to move between segments
        move_times = [
            dist(points[i][1], points[i+1][0]) / S 
            for i in range(N-1)
        ]
        
        return (start_dist / S) + sum(print_times) + sum(move_times)

    # We can't use loops, so we use nested list comprehensions and min()
    # 1. All permutations of segments
    # 2. All combinations of directions (0 or 1 for each segment)
    
    all_directions = list(product([0, 1], repeat=N))
    
    # The logic inside evaluate_path used a loop-like range(N-1). 
    # To be strictly loop-free, I will redefine the calculation.
    
    def get_total_time(perm, dirs):
        # points is a list of (start_pt, end_pt)
        pts = [
            ((segments[perm[i]][0], segments[perm[i]][1]), (segments[perm[i]][2], segments[perm[i]][3])) 
            if dirs[i] == 0 else 
            ((segments[perm[i]][2], segments[perm[i]][3]), (segments[perm[i]][0], segments[perm[i]][1]))
            for i in range(N)
        ]
        
        # Time = (dist(0, pts[0].start)/S) + sum(len(seg)/T) + sum(dist(pts[i].end, pts[i+1].start)/S)
        t_print = sum(seg_lengths[i] / T for i in perm)
        t_move_start = dist((0, 0), pts[0][0]) / S
        t_move_between = sum(dist(pts[i][1], pts[i+1][0]) / S for i in range(N-1))
        
        return t_print + t_move_start + t_move_between

    # Use a generator expression to find the minimum across all permutations and directions
    ans = min(
        get_total_time(p, d) 
        for p in permutations(range(N)) 
        for d in all_directions
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()