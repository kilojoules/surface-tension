import itertools
import math
from functools import reduce

def solve():
    # Read N, S, T
    try:
        line1 = input().split()
        if not line1: return
        N, S, T = map(int, line1)
        
        # Read segments
        segs = [list(map(int, input().split())) for _ in range(N)]
    except EOFError:
        return

    # Pre-calculate lengths of segments and time to print them
    # seg_data[i] = (point1, point2, print_time)
    seg_data = [
        ((s[0], s[1]), (s[2], s[3]), 
         math.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2) / T) 
        for s in segs
    ]

    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to try all permutations of segments and all directions for each segment.
    # A direction is represented by 0 (p1 -> p2) or 1 (p2 -> p1).
    
    # Generate all permutations of segment indices
    perms = itertools.permutations(range(N))
    # Generate all possible direction combinations (2^N)
    dirs = itertools.product([0, 1], repeat=N)
    
    # To avoid loops, we use a generator expression inside min()
    # For a fixed permutation 'p' and direction 'd':
    # We calculate the total time by iterating through the sequence.
    
    # We use a helper function to calculate the cost of a specific sequence
    def calculate_cost(p, d):
        # current_pos starts at (0, 0)
        # state: (current_pos, total_time)
        
        def step(state, idx):
            curr_pos, total_time = state
            seg_idx = p[idx]
            p1, p2, t_print = seg_data[seg_idx]
            
            # Determine start and end points based on direction d[seg_idx]
            start_pt = p1 if d[seg_idx] == 0 else p2
            end_pt = p2 if d[seg_idx] == 0 else p1
            
            # Time = move to start + print to end
            move_time = dist(curr_pos, start_pt) / S
            return (end_pt, total_time + move_time + t_print)

        final_state = reduce(step, range(N), ((0, 0), 0.0))
        return final_state[1]

    # We cannot use 'for' loops, so we use nested generator expressions.
    # We evaluate all permutations and all direction combinations.
    # Since N is small (<= 6), N! * 2^N is at most 720 * 64 = 46,080.
    
    ans = min(
        calculate_cost(p, d)
        for p in perms
        for d in dirs
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()