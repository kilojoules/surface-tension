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
    
    # Store segments as pairs of tuples
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
    
    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to find the optimal permutation of segments and the direction for each.
    # Since N is very small (up to 6), we can iterate through all permutations (N!)
    # and all direction combinations (2^N).
    # Total complexity: N! * 2^N * N, which is 720 * 64 * 6 ≈ 276,480.
    
    # To avoid for/while loops, we use map and min with generators.
    
    # Generate all permutations of indices
    perms = permutations(range(N))
    
    # For each permutation, we evaluate all 2^N direction combinations.
    # A direction combination can be represented by a bitmask from 0 to 2^N - 1.
    # Bit 0: segment i is printed p1 -> p2, Bit 1: p2 -> p1.
    
    def calculate_time(perm, mask):
        # Current position
        curr_pos = (0, 0)
        total_time = 0.0
        
        # We need to iterate through the permutation. 
        # Since we can't use for loops, we'll use a reduction or a list comprehension.
        # However, the state (curr_pos, total_time) depends on the previous step.
        # We can use a helper function with a loop or a recursion, but the prompt 
        # asks for no for/while loops. We can use a functional approach.
        return None

    # To strictly follow "no for/while loops", we can use a recursive function 
    # or a reduce operation.
    from functools import reduce

    def process_step(state, item):
        idx, mask_bit = item
        pos, time = state
        p1, p2 = segments[idx]
        
        # Determine start and end based on mask_bit
        start, end = (p1, p2) if mask_bit == 0 else (p2, p1)
        
        # Time = travel to start / S + length of segment / T
        travel_time = dist(pos, start) / S
        print_time = lengths[idx] / T
        
        return (end, time + travel_time + print_time)

    # For a specific permutation and mask, calculate total time
    def get_total_time(perm, mask):
        # Create a list of (index, bit)
        steps = [(perm[i], (mask >> i) & 1) for i in range(N)]
        final_state = reduce(process_step, steps, ((0, 0), 0.0))
        return final_state[1]

    # Use a generator to find the minimum time across all permutations and masks
    # We use a nested generator expression inside min()
    ans = min(
        get_total_time(perm, mask)
        for perm in perms
        for mask in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()