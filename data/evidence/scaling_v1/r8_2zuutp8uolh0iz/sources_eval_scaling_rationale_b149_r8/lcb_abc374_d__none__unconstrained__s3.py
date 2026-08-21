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

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute lengths of segments for T-speed printing
    seg_lengths = [dist(s[0], s[1]) for s in segments]

    # We need to try all permutations of segments and all 2^N orientations
    # A state in our reduce will be (current_pos, total_time)
    
    def calculate_total_time(perm_indices, orientations):
        # perm_indices: order of segments to visit
        # orientations: 0 means start at s[0] end at s[1], 1 means vice versa
        
        def folder(state, item):
            curr_pos, curr_time = state
            seg_idx, orient = item
            
            p1, p2 = segments[seg_idx]
            start_node = p1 if orient == 0 else p2
            end_node = p2 if orient == 0 else p1
            
            # Time to move to start_node (at speed S) + time to print (at speed T)
            move_time = dist(curr_pos, start_node) / S
            print_time = seg_lengths[seg_idx] / T
            
            return (end_node, curr_time + move_time + print_time)

        # Zip the permutation with the chosen orientations
        sequence = list(zip(perm_indices, orientations))
        final_state = reduce(folder, sequence, ((0, 0), 0.0))
        return final_state[1]

    # Generate all permutations of indices
    all_perms = permutations(range(N))
    # Generate all possible orientation combinations (0 or 1 for each segment)
    all_orients = list(product([0, 1], repeat=N))

    # We need to map orientations to the specific segments in the permutation.
    # Since orientations are provided as a fixed tuple of length N, 
    # we can just iterate through all 2^N combinations and all N! permutations.
    
    # To avoid nested loops, we use a generator expression inside min()
    # We evaluate: for each permutation P, and each orientation set O, 
    # the cost is calculate_total_time(P, O)
    # Note: O must be aligned with the permutation. 
    # Actually, it's simpler: for a fixed permutation, there are 2^N ways to 
    # orient the segments. The orientation of the i-th segment in the 
    # permutation is what matters.
    
    ans = min(
        calculate_total_time(p, o)
        for p in all_perms
        for o in all_orients
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()