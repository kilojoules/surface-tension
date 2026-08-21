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

    def get_dist(start, end, obstacle, n):
        # There are two ways to move on a ring: clockwise and counter-clockwise.
        # One way is (start + x) % n, the other is (start - x) % n.
        # We need to check if the obstacle is in the path.
        
        # Normalize to 0-indexed
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Clockwise distance
        cw_dist = (e - s) % n
        # The obstacle is hit if it lies between s and e clockwise.
        # Obstacle is at relative position (o - s) % n.
        # It blocks if 0 < (o - s) % n < cw_dist.
        cw_blocked = 0 < (o - s) % n < cw_dist
        
        # Counter-clockwise distance
        ccw_dist = (s - e) % n
        # It blocks if 0 < (s - o) % n < ccw_dist.
        ccw_blocked = 0 < (s - o) % n < ccw_dist
        
        # We are guaranteed the instruction is achievable.
        # Return the minimum of the non-blocked paths.
        res = []
        if not cw_blocked: res.append(cw_dist)
        if not ccw_blocked: res.append(ccw_dist)
        return min(res)

    # State: (left_pos, right_pos, total_ops)
    # Initial state: Left=1, Right=2, Ops=0
    initial_state = (1, 2, 0)

    def step(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand stays at r
            dist = get_dist(l, t, r, N)
            return (t, r, total + dist)
        else:
            # Move right hand to t, left hand stays at l
            dist = get_dist(r, t, l, N)
            return (l, t, total + dist)

    final_state = reduce(step, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()