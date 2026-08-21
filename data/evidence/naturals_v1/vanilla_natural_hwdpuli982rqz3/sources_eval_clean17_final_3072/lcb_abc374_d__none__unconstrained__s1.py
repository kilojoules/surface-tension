import sys
from itertools import permutations
import math

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Store segments as pairs of points
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]
    
    # Precompute lengths of segments
    lengths = [
        math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
        for p1, p2 in segments
    ]
    
    # Distance helper
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to find the optimal order of segments and the direction of each segment.
    # Since N is very small (up to 6), we can iterate through all permutations of segments
    # and all 2^N combinations of directions.
    
    # To avoid for/while loops, we use map and min with generators.
    
    # Generate all permutations of indices (0, 1, ..., N-1)
    all_perms = permutations(range(N))
    
    # For a given permutation and a bitmask representing direction (0: p1->p2, 1: p2->p1)
    # Calculate total time.
    def calculate_time(perm, mask):
        # Current position
        curr_pos = (0, 0)
        total_time = 0.0
        
        # We need to simulate the process. Since we can't use for loops, 
        # we can use a reduce-like approach or a list comprehension with a state.
        # However, the prompt allows "complete Python program", and usually, 
        # "no for/while loops" is a specific constraint for some challenges.
        # Re-reading: The prompt does NOT forbid loops. It just asks for the program.
        
        # Let's use a loop for clarity and correctness.
        time = 0.0
        pos = (0, 0)
        for i in perm:
            p1, p2 = segments[i]
            # Determine start and end based on mask
            # mask is an integer, we check the i-th bit? No, the bit corresponding to the segment in the perm.
            # Let's use a different approach for the mask.
            pass

    # Since I need to avoid loops if I were strictly following "functional" styles, 
    # but loops are standard Python, I will use them.
    
    # To handle the 2^N directions without nested loops, I'll use a recursive-like 
    # structure or just standard loops which are permitted in "complete Python programs".
    
    # Let's use a helper to calculate time for a specific sequence of directed edges.
    def get_total_time(sequence):
        # sequence is a list of (start_node, end_node, length)
        # Total time = sum(dist(prev_end, curr_start)/S + length/T)
        
        # Zip the sequence with itself shifted to get (prev_end, curr_start)
        # The first start is relative to (0,0)
        starts = [s[0] for s in sequence]
        ends = [s[1] for s in sequence]
        lens = [s[2] for s in sequence]
        
        # Distance from (0,0) to first start
        first_move = dist((0,0), starts[0]) / S
        
        # Distances between segments
        middle_moves = sum([dist(ends[i], starts[i+1]) / S for i in range(N-1)])
        
        # Printing times
        print_times = sum([l / T for l in lens])
        
        return first_move + middle_moves + print_times

    # Generate all possible sequences of directed segments
    # For each permutation of segments, there are 2^N ways to orient them.
    
    # We can represent orientations as a tuple of booleans
    from itertools import product
    all_orientations = product([False, True], repeat=N)
    
    # Use a generator to find the minimum time across all permutations and orientations
    ans = min(
        get_total_time([
            (segments[i][0] if orient else segments[i][1], 
             segments[i][1] if orient else segments[i][0], 
             lengths[i])
            for i, orient in zip(perm, orient_mask)
        ])
        for perm in permutations(range(N))
        for orient_mask in product([False, True], repeat=N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()