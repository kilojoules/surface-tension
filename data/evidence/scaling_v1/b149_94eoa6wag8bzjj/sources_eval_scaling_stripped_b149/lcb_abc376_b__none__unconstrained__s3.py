import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate shortest distance between two points on a ring of size N
    # without passing through a forbidden point 'obstacle'
    def get_dist(start, end, obstacle):
        if start == end:
            return 0
        
        # There are two paths on a ring: clockwise and counter-clockwise
        # Path 1: start -> start+1 -> ... -> end (modulo N)
        # Path 2: start -> start-1 -> ... -> end (modulo N)
        
        # To check if a path is blocked, we check if the obstacle lies 
        # strictly between start and end on that arc.
        
        # Normalize coordinates to 0...N-1 for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Clockwise distance
        dist_cw = (e - s) % N
        # The obstacle is on the clockwise path if (o - s) % N < dist_cw
        blocked_cw = (o - s) % N < dist_cw
        
        # Counter-clockwise distance
        dist_ccw = (s - e) % N
        # The obstacle is on the counter-clockwise path if (o - e) % N < dist_ccw
        blocked_ccw = (o - e) % N < dist_ccw
        
        # We are guaranteed the instruction is achievable, so at least one path is open.
        # We want the minimum of the available paths.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        
        return min(res)

    # State: (current_l, current_r, total_dist)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is the obstacle
            d = get_dist(l, t, r)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l is the obstacle
            d = get_dist(r, t, l)
            return (l, t, total + d)

    # Use reduce to simulate the sequence of instructions
    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()