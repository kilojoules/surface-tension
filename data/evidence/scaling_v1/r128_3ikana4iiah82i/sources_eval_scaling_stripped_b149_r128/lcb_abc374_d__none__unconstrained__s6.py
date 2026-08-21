import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples ((x1, y1), (x2, y2))
    segments = [
        ((int(input_data[3 + i*4]), int(input_data[4 + i*4])), 
         (int(input_data[5 + i*4]), int(input_data[6 + i*4])))
        for i in range(N)
    ]

    # Distance helper function
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # For a given permutation of segments, we need to decide the direction of each segment.
    # Since N is small (<= 6), we can iterate through all 2^N direction combinations.
    # However, it's cleaner to use a list comprehension to evaluate all direction combinations.
    
    # We generate all possible orderings of segments
    all_orders = permutations(segments)
    
    # For each order, we evaluate all 2^N direction combinations.
    # A direction combination can be represented as a tuple of 0s and 1s.
    # If dir_bit == 0, we go from p1 to p2. If 1, from p2 to p1.
    
    # To avoid explicit loops, we use a nested comprehension:
    # 1. Permute the segments.
    # 2. For each permutation, generate all binary strings of length N.
    # 3. For each binary string, calculate the total time.
    
    # We use a helper to calculate the time for a specific sequence of directed segments.
    # A directed segment is (start_point, end_point).
    
    def calc_time(directed_segs):
        # The laser starts at (0, 0)
        start_pos = (0, 0)
        
        # We need to calculate the sum of:
        # (dist(current_pos, seg_start) / S) + (dist(seg_start, seg_end) / T)
        # We use reduce to keep track of the (current_total_time, last_position)
        
        final_state = reduce(
            lambda state, seg: (
                state[0] + dist(state[1], seg[0]) / S + dist(seg[0], seg[1]) / T,
                seg[1]
            ),
            directed_segs,
            (0.0, start_pos)
        )
        return final_state[0]

    # Generate all possible direction combinations as a list of tuples
    # Since N <= 6, 2^6 = 64. We can pre-calculate these.
    directions = [
        tuple((0 if (i >> j) & 1 == 0 else 1) for j in range(N))
        for i in range(1 << N)
    ]

    # The core logic:
    # For every permutation of segments, and every combination of directions,
    # create the list of directed segments and pass it to calc_time.
    
    ans = min(
        calc_time([
            (seg[dir_bit] if dir_bit == 0 else seg[1], 
             seg[1] if dir_bit == 0 else seg[0])
            for seg, dir_bit in zip(order, dirs)
        ])
        for order in all_orders
        for dirs in directions
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()