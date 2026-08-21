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

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # For a given permutation of segments, we need to decide the direction of each segment.
    # There are 2^N possible direction combinations.
    # We can use a list comprehension to evaluate all direction combinations for a permutation.
    # Each direction combination is represented by a tuple of 0s and 1s.
    
    # To avoid explicit loops, we use a generator expression inside min()
    # We iterate through all permutations of the segments.
    # For each permutation, we iterate through all 2^N direction assignments.
    
    # We use a list of all possible direction bitmasks
    masks = [tuple((i >> j) & 1 for j in range(N)) for i in range(1 << N)]
    
    # The core logic to calculate time for a specific order and direction set:
    # We use reduce to keep track of (current_position, total_time)
    def calc_time(order, mask):
        # Determine the start and end points for each segment based on the mask
        # If mask[i] == 0: start is p1, end is p2. If 1: start is p2, end is p1.
        oriented = [
            (seg[0], seg[1]) if m == 0 else (seg[1], seg[0])
            for seg, m in zip(order, mask)
        ]
        
        # Initial state: laser at (0, 0), time 0
        initial_state = ((0, 0), 0.0)
        
        # Process each oriented segment
        final_state = reduce(
            lambda state, seg: (
                seg[1], 
                state[1] + dist(state[0], seg[0]) / S + dist(seg[0], seg[1]) / T
            ),
            oriented,
            initial_state
        )
        return final_state[1]

    # We find the minimum time across all permutations and all direction masks.
    # Using a generator expression inside min() satisfies the "no loop" constraint.
    result = min(
        calc_time(p, m)
        for p in permutations(segments)
        for m in masks
    )
    
    print(f"{result:.20f}")

if __name__ == "__main__":
    solve()