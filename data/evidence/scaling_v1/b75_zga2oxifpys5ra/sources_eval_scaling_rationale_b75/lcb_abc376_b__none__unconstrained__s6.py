import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T) tuples
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Helper to calculate distance between start and end on a ring of size N
    # avoiding the 'blocked' node.
    # Since we can only move one hand, the other hand acts as a wall.
    # There are only two directions on a ring. One must be clear.
    def get_dist(start, end, blocked, n):
        # Clockwise distance (increasing index)
        # Path: start -> start+1 -> ... -> end
        # The blocked node is encountered if it lies between start and end
        # in the direction of movement.
        
        # Normalize coordinates to 0..N-1 for easier modulo arithmetic
        s, e, b = start - 1, end - 1, blocked - 1
        
        # Distance clockwise
        dist_cw = (e - s) % n
        # Check if blocked node is in the clockwise path
        # Blocked is in path if (b - s) % n < dist_cw
        is_blocked_cw = (b - s) % n < dist_cw
        
        # Distance counter-clockwise
        dist_ccw = (s - e) % n
        # Blocked is in path if (s - b) % n < dist_ccw
        is_blocked_ccw = (s - b) % n < dist_ccw
        
        # We are guaranteed the instruction is achievable, 
        # so at least one path is clear.
        if is_blocked_cw:
            return dist_ccw
        if is_blocked_ccw:
            return dist_cw
        # If neither is blocked (only possible if N > 2 and start/end/blocked are distinct),
        # return the minimum of the two.
        return min(dist_cw, dist_ccw)

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)
    
    # Use reduce to process instructions and accumulate distance
    final_state = reduce(
        lambda state, instr: (
            # New state tuple: (new_l, new_r, new_dist)
            (
                instr[1] if instr[0] == 'L' else state[0], # New L
                instr[1] if instr[0] == 'R' else state[1], # New R
                state[2] + (
                    get_dist(state[0], instr[1], state[1], N) 
                    if instr[0] == 'L' 
                    else get_dist(state[1], instr[1], state[0], N)
                )
            )
        ),
        instructions,
        initial_state
    )
    
    print(final_state[2])

if __name__ == "__main__":
    solve()