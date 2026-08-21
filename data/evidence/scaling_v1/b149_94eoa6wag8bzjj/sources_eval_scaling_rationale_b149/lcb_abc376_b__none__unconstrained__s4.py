import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance on a ring with a blockage
    # current: position of hand moving
    # target: destination
    # other: position of the hand that cannot be moved
    def get_dist(current, target, other, n):
        # The ring is 1-indexed. Convert to 0-indexed for easier modulo math.
        c, t, o = current - 1, target - 1, other - 1
        
        # There are two ways to move on the ring: clockwise and counter-clockwise.
        # One direction is blocked if 'other' lies on the path between 'current' and 'target'.
        
        # Clockwise distance
        dist_cw = (t - c) % n
        # Check if 'other' is encountered moving clockwise from c to t
        # 'other' is on the path if (o - c) % n < dist_cw
        blocked_cw = (o - c) % n < dist_cw
        
        # Counter-clockwise distance
        dist_ccw = (c - t) % n
        # Check if 'other' is encountered moving counter-clockwise from c to t
        # 'other' is on the path if (c - o) % n < dist_ccw
        blocked_ccw = (c - o) % n < dist_ccw
        
        # We want the minimum distance among the non-blocked paths.
        # The problem guarantees the instruction is achievable.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res)

    # State: (left_pos, right_pos, total_dist)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def reducer(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(reducer, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()