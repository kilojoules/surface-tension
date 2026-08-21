import sys
import math
from itertools import permutations, product
from functools import reduce

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

    # Precompute lengths of segments to avoid redundant calculations
    # length_i = sqrt((Cx-Ax)^2 + (Dy-By)^2)
    seg_lengths = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2)
        for s in segments
    ]

    # We need to try all permutations of segments and all directions (2^N)
    # A direction is represented by 0 (Start -> End) or 1 (End -> Start)
    
    # Generate all permutations of indices 0 to N-1
    all_perms = permutations(range(N))
    # Generate all possible direction combinations (0 or 1 for each segment)
    all_dirs = product([0, 1], repeat=N)

    def get_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_total_time(perm, dirs):
        # Current state: (current_x, current_y, accumulated_time)
        # We use reduce to simulate the traversal through the chosen sequence
        
        def folder(state, idx):
            curr_pos, total_time = state
            seg_idx = perm[idx]
            direction = dirs[seg_idx]
            
            # Determine start and end points based on direction
            p_start = segments[seg_idx][0] if direction == 0 else segments[seg_idx][1]
            p_end = segments[seg_idx][1] if direction == 0 else segments[seg_idx][0]
            
            # Time to move to start point (at speed S) + time to print (at speed T)
            move_time = get_dist(curr_pos, p_start) / S
            print_time = seg_lengths[seg_idx] / T
            
            return (p_end, total_time + move_time + print_time)

        # Initial state: position (0,0), time 0
        final_state = reduce(folder, range(N), ((0, 0), 0.0))
        return final_state[1]

    # Since we cannot use loops, we use list comprehensions and min()
    # We need to iterate over all permutations and all direction combinations.
    # Note: dirs is mapped to the segment index, so we use a fixed dir-tuple 
    # and access it via the permutation.
    
    # To avoid nested loops, we flatten the search space into one generator
    # We use a comprehension to evaluate all possibilities.
    # Because N is small (<= 6), N! * 2^N is at most 720 * 64 = 46,080.
    
    # We need to redefine how dirs are handled to fit the flat structure.
    # Let's use a product of (perm, dir_tuple)
    
    results = [
        calculate_total_time(p, d) 
        for p in permutations(range(N)) 
        for d in product([0, 1], repeat=N)
    ]
    
    print(f"{min(results):.20f}")

if __name__ == "__main__":
    solve()