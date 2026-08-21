import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    instructions = []
    for i in range(Q):
        instructions.append((input_data[2 + 2*i], int(input_data[3 + 2*i])))

    # Helper to calculate shortest distance on ring avoiding a blocked node
    # s: start, t: target, x: blocked node
    def get_dist(s, t, x):
        if s == t:
            return 0
        
        # Clockwise distance (s -> s+1 -> ... -> t)
        # The path is s, (s+1)%N, ..., t. 
        # It is blocked if x is any node in the sequence between s and t.
        # To simplify, we check if x is "between" s and t clockwise.
        
        # Normalize to 0-indexed
        s_0, t_0, x_0 = s-1, t-1, x-1
        
        def is_between_cw(start, end, block):
            # Check if block is encountered when going clockwise from start to end
            if start == end: return False
            if start < end:
                return start < block < end
            else: # wrap around
                return block > start or block < end

        # Distance clockwise
        dist_cw = (t_0 - s_0) % N
        # Distance counter-clockwise
        dist_ccw = (s_0 - t_0) % N
        
        # Check if paths are blocked
        blocked_cw = is_between_cw(s_0, t_0, x_0)
        blocked_ccw = is_between_cw(t_0, s_0, x_0)
        
        # We are guaranteed the target is reachable.
        # If clockwise is blocked, must go counter-clockwise.
        # If counter-clockwise is blocked, must go clockwise.
        # If neither is blocked, take the minimum.
        # If both are blocked, this case is guaranteed not to happen by problem statement.
        
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res)

    # State: (current_l, current_r, total_dist)
    # Initial state: L=1, R=2, dist=0
    def own_reduce(state, instr):
        l, r, d = state
        h, t = instr
        if h == 'L':
            return (t, r, d + get_dist(l, t, r))
        else:
            return (l, t, d + get_dist(r, t, l))

    final_state = reduce(own_reduce, instructions, (1, 2, 0))
    print(final_state[2])

if __name__ == "__main__":
    solve()