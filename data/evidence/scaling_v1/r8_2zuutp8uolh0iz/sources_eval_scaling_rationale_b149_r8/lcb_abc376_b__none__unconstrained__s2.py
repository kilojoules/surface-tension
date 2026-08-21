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

    # Helper to calculate the shortest distance on a ring of size N
    # from start to end, avoiding the obstacle 'obs'.
    # Since we can only move one hand, the other hand (obs) 
    # blocks one of the two possible directions (clockwise or counter-clockwise).
    # There is only one valid path if the obstacle is in the way.
    def get_dist(start, end, obs, n):
        if start == end:
            return 0
        
        # There are two directions: 
        # 1. Increasing indices (1 -> 2 -> ... -> N -> 1)
        # 2. Decreasing indices (1 -> N -> ... -> 2 -> 1)
        
        # Check if the path is blocked by the obstacle.
        # A path is blocked if the obstacle lies between start and end.
        
        # Normalize to 0-indexed for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obs - 1
        
        # Clockwise distance
        cw_dist = (e - s) % n
        # Counter-clockwise distance
        ccw_dist = (s - e) % n
        
        # The obstacle blocks the clockwise path if it's "between" s and e clockwise.
        # The obstacle is at index o. It blocks clockwise if (o-s)%n < cw_dist.
        # However, the problem says we can't move TO the destination part if the other hand is there.
        # But the guarantee says T_i != other_hand.
        # The only way a path is blocked is if the obstacle is strictly between start and end.
        
        # Clockwise path: s, (s+1)%n, ..., e
        # Blocked if there exists k such that 0 < k < cw_dist and (s+k)%n == o
        cw_blocked = (o - s) % n < cw_dist and (o - s) % n != 0
        
        # Counter-clockwise path: s, (s-1)%n, ..., e
        # Blocked if there exists k such that 0 < k < ccw_dist and (s-k)%n == o
        ccw_blocked = (s - o) % n < ccw_dist and (s - o) % n != 0
        
        # Since the problem guarantees the instruction is achievable,
        # at least one path must be open.
        res = []
        if not cw_blocked: res.append(cw_dist)
        if not ccw_blocked: res.append(ccw_dist)
        
        return min(res)

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)
    
    def transition(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()