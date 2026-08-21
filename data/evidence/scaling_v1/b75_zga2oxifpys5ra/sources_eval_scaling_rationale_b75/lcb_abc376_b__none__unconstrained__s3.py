import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as a list of (H, T)
    instructions = []
    for i in range(Q):
        instructions.append((input_data[2 + 2*i], int(input_data[3 + 2*i])))

    # Helper to calculate distance between start and end on a ring of size N
    # avoiding the 'blocked' vertex.
    # The movement is restricted: you cannot step on the blocked vertex.
    def get_dist(start, end, blocked, n):
        if start == end:
            return 0
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # Path 1: start -> start+1 -> ... -> end (modulo N)
        # Path 2: start -> start-1 -> ... -> end (modulo N)
        
        # Clockwise distance
        dist_cw = (end - start) % n
        # Counter-clockwise distance
        dist_ccw = (start - end) % n
        
        # Check if blocked vertex is on the clockwise path
        # The blocked vertex is on the CW path if (blocked - start) % n < dist_cw
        # Note: the destination 'end' is guaranteed not to be 'blocked'.
        blocked_on_cw = (blocked - start) % n < dist_cw
        blocked_on_ccw = (start - blocked) % n < dist_ccw
        
        # We can only take the path if the blocked vertex is not on it.
        # Since the problem guarantees the instruction is achievable,
        # at least one path must be clear.
        res = []
        if not blocked_on_cw:
            res.append(dist_cw)
        if not blocked_on_ccw:
            res.append(dist_ccw)
            
        return min(res)

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    # We use 0-indexed coordinates internally for easier modulo math
    initial_state = (0, 1, 0) 

    def step(state, instr):
        l, r, total = state
        h, t = instr
        t_idx = t - 1 # 0-indexed
        
        if h == 'L':
            # Move left hand to t_idx, right hand (r) is blocked
            d = get_dist(l, t_idx, r, N)
            return (t_idx, r, total + d)
        else:
            # Move right hand to t_idx, left hand (l) is blocked
            d = get_dist(r, t_idx, l, N)
            return (l, t_idx, total + d)

    final_state = reduce(step, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()