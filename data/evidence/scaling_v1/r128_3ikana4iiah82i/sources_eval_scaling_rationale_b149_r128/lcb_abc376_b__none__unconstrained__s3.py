import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H_i, T_i)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate shortest distance on ring with a blocked node
    # l, r: current positions; target: destination; 
    # blocked: the position of the hand that must stay still.
    def get_dist(start, target, blocked, n):
        if start == target:
            return 0
        
        # There are two directions to move on the ring: clockwise and counter-clockwise.
        # A path is valid if the 'blocked' node is not encountered.
        
        # Clockwise distance
        cw_dist = (target - start + n) % n
        # Check if blocked node is in the clockwise path
        # The blocked node is in the path if (blocked - start + n) % n < cw_dist
        cw_blocked = (blocked - start + n) % n < cw_dist
        
        # Counter-clockwise distance
        ccw_dist = (start - target + n) % n
        # Check if blocked node is in the counter-clockwise path
        ccw_blocked = (start - blocked + n) % n < ccw_dist
        
        # We are guaranteed the instruction is achievable, 
        # so at least one path must be clear.
        if not cw_blocked and not ccw_blocked:
            return min(cw_dist, ccw_dist)
        if not cw_blocked:
            return cw_dist
        if not ccw_blocked:
            return ccw_dist
        return float('inf') # Should not happen per guarantees

    # State: (left_pos, right_pos, total_distance)
    # Initial state: Left hand at 1, Right hand at 2
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, dist = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is blocked
            d = get_dist(l, t, r, N)
            return (t, r, dist + d)
        else:
            # Move right hand to t, left hand l is blocked
            d = get_dist(r, t, l, N)
            return (l, t, dist + d)

    # Use reduce to process all instructions and accumulate distance
    final_state = reduce(transition, instructions, initial_state)
    
    # The result is the third element of the final state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()