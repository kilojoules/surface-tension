import sys
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
    
    segments = [
        (int(input_data[3 + 2*i]), int(input_data[4 + 2*i]), 
         int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        for i in range(N)
    ]

    # Pre-calculate lengths of segments
    # length = sqrt((x2-x1)^2 + (y2-y1)^2)
    seg_lengths = [
        ((s[2]-s[0])**2 + (s[3]-s[1])**2)**0.5 
        for s in segments
    ]

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    # We need to try all permutations of segments and all directions for each segment.
    # A state is (current_x, current_y, total_time)
    # We use reduce to iterate through the chosen sequence of segments.
    
    # Generate all possible orderings of segments
    all_orders = permutations(range(N))
    
    # Generate all possible direction choices (0: start->end, 1: end->start)
    all_directions = product([0, 1], repeat=N)

    # To avoid loops, we can't use 'all_directions' inside 'all_orders' without 
    # a comprehension that would be too large. 
    # Instead, we can structure the problem as:
    # For each permutation, we want to find the min time.
    # Since N is small (6), we can use a recursive-like approach via reduce or 
    # simply map over the permutations and then over the directions.
    
    def evaluate_path(order, dirs):
        # Initial position
        start_pos = (0, 0)
        
        # We need to track (current_pos, current_time)
        # The sequence of actions is: 
        # For each segment in order:
        #   1. Move to the chosen start point (speed S)
        #   2. Move to the chosen end point (speed T)
        
        def step(state, item):
            curr_pos, curr_time = state
            seg_idx, direction = item
            
            p1 = (segments[seg_idx][0], segments[seg_idx][1])
            p2 = (segments[seg_idx][2], segments[seg_idx][3])
            
            # Determine start and end points based on direction
            s_pt, e_pt = (p1, p2) if direction == 0 else (p2, p1)
            
            # Time to move to start point + time to print segment
            travel_time = dist(curr_pos, s_pt) / S
            print_time = seg_lengths[seg_idx] / T
            
            return (e_pt, curr_time + travel_time + print_time)

        # Zip the order and directions together
        final_state = reduce(step, zip(order, dirs), (start_pos, 0.0))
        return final_state[1]

    # Since we cannot use for-loops, we use map/min/comprehensions.
    # We evaluate all permutations and all direction combinations.
    # Total complexity: N! * 2^N * N
    # For N=6: 720 * 64 * 6 approx 276,480 operations.
    
    # We use a nested comprehension to find the minimum time.
    # Note: The constraint says no for-loops. 
    # We can use a generator expression inside min().
    
    ans = min(
        evaluate_path(order, dirs)
        for order in all_orders
        for dirs in all_directions
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()