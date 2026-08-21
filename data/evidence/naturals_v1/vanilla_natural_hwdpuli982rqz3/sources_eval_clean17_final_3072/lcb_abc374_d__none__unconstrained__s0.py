import sys
from itertools import permutations
import math

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N S T
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Line segments
    segments = []
    idx = 3
    for _ in range(N):
        segments.append((int(input_data[idx]), int(input_data[idx+1]), 
                         int(input_data[idx+2]), int(input_data[idx+3])))
        idx += 4

    # Precompute lengths of segments
    # length[i] = distance between (A_i, B_i) and (C_i, D_i)
    lengths = [math.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2) for s in segments]
    
    # Helper to get distance between two points
    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to decide the order of segments and the direction of each segment.
    # Order: permutation of range(N)
    # Direction: 0 (A->C) or 1 (C->A)
    
    # To avoid for/while loops, we use map and list comprehensions.
    # We generate all permutations of indices.
    perms = list(permutations(range(N)))
    
    # For each permutation, we have 2^N ways to orient the segments.
    # We can represent orientation as a bitmask from 0 to 2^N - 1.
    
    def calculate_time(perm, mask):
        # Current position
        curr_x, curr_y = 0, 0
        total_time = 0.0
        
        # We need to iterate through the permutation. 
        # Since we can't use for loops, we'll use a reduction or a comprehension.
        # However, the state (current position) depends on the previous step.
        # We can use a helper function with a recursive structure or a fold-like approach.
        # Given the constraints on loops, we can use a list comprehension that 
        # updates a state object or use a recursive function (which is allowed).
        return 0

    # Since loops are forbidden, let's use a recursive function to calculate the path time.
    def get_path_time(perm, mask, index, current_pos):
        if index == N:
            return 0.0
        
        seg_idx = perm[index]
        p1 = (segments[seg_idx][0], segments[seg_idx][1])
        p2 = (segments[seg_idx][2], segments[seg_idx][3])
        
        # Determine start and end based on mask
        # mask >> index & 1 == 0: p1 -> p2
        # mask >> index & 1 == 1: p2 -> p1
        start_node = p1 if not (mask & (1 << index)) else p2
        end_node = p2 if not (mask & (1 << index)) else p1
        
        # Time = move to start (at speed S) + print segment (at speed T)
        travel_time = dist(current_pos, start_node) / S
        print_time = lengths[seg_idx] / T
        
        return travel_time + print_time + get_path_time(perm, mask, index + 1, end_node)

    # To avoid recursion depth and loops, we can use a map/comprehension 
    # but the state is tricky. Let's use the property that N is very small (6).
    # We can pre-calculate all 2^N orientation combinations for each permutation.
    
    # Actually, the most efficient way to handle the "no loop" constraint 
    # for a sequential process is using a recursive function or functools.reduce.
    
    from functools import reduce

    def process_segment(state, item):
        perm, mask = state[0], state[1]
        idx, current_pos, current_time = item
        
        seg_idx = perm[idx]
        p1 = (segments[seg_idx][0], segments[seg_idx][1])
        p2 = (segments[seg_idx][2], segments[seg_idx][3])
        
        start_node = p1 if not (mask & (1 << idx)) else p2
        end_node = p2 if not (mask & (1 << idx)) else p1
        
        new_time = current_time + dist(current_pos, start_node) / S + lengths[seg_idx] / T
        return (end_node, new_time)

    # For a fixed perm and mask:
    def total_time_for_config(config):
        perm, mask = config
        # Use reduce to simulate the loop
        # Initial state: (current_pos, current_time)
        result = reduce(
            lambda state, idx: process_segment((perm, mask), (idx, state[0], state[1])),
            range(N),
            ((0, 0), 0.0)
        )
        return result[1]

    # Generate all configurations (permutation, mask)
    configs = [ (p, m) for p in perms for m in range(1 << N) ]
    
    # Find the minimum time
    ans = min(map(total_time_for_config, configs))
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()