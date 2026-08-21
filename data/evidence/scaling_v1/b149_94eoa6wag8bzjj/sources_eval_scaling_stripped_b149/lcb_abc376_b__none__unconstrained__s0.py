import sys
from functools import reduce

def solve():
    # Read input
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

    def get_dist(start, end, obstacle, n):
        """
        Calculates the shortest distance from start to end on a ring of size n,
        given that the 'obstacle' part cannot be stepped on.
        Returns infinity if the destination is unreachable.
        """
        # There are two directions to move on a ring: clockwise and counter-clockwise.
        # We check if the obstacle lies in the path for either direction.
        
        # Normalize coordinates to 0-indexed for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Clockwise distance
        cw_dist = (e - s) % n
        # The obstacle is in the clockwise path if its relative position from s 
        # is less than the distance to e.
        cw_blocked = (o - s) % n < cw_dist
        
        # Counter-clockwise distance
        ccw_dist = (s - e) % n
        # The obstacle is in the counter-clockwise path if its relative position 
        # from s (going backwards) is less than the distance to e.
        ccw_blocked = (s - o) % n < ccw_dist
        
        # Return the minimum of the unblocked paths
        res = []
        if not cw_blocked: res.append(cw_dist)
        if not ccw_blocked: res.append(ccw_dist)
        
        return min(res) if res else float('inf')

    def process_instruction(state, instr):
        """
        state: (current_l, current_r, total_ops)
        instr: (H, T)
        """
        l, r, ops = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is the obstacle
            dist = get_dist(l, t, r, N)
            return (t, r, ops + dist)
        else:
            # Move right hand to t, left hand l is the obstacle
            dist = get_dist(r, t, l, N)
            return (l, t, ops + dist)

    # Initial state: Left hand at 1, Right hand at 2, 0 operations
    initial_state = (1, 2, 0)
    
    # Use reduce to iterate through instructions and accumulate the state
    final_state = reduce(process_instruction, instructions, initial_state)
    
    # The result is the third element of the final state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()