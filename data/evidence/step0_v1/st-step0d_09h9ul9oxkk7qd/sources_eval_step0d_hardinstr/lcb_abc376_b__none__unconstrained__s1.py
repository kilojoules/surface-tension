import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    instructions = []
    for i in range(Q):
        instructions.append((input_data[2 + 2*i], int(input_data[3 + 2*i])))

    # Helper to calculate distance between start and end on a ring of size N
    # given that the other hand is at position 'obs'
    # The movement must not pass through 'obs'
    def get_dist(start, end, obs, n):
        if start == end:
            return 0
        
        # There are two directions on the ring:
        # 1. Increasing (start -> start+1 -> ...)
        # 2. Decreasing (start -> start-1 -> ...)
        
        # Check if the path is blocked by the obstacle
        # A path is blocked if the obstacle lies between start and end
        # We normalize coordinates to 0...N-1 for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obs - 1
        
        # Clockwise distance (increasing index)
        # The path is s, (s+1)%n, (s+2)%n ... e
        # It is blocked if o is encountered before e
        # The number of steps is (e - s) % n
        dist_cw = (e - s) % n
        # The obstacle is at (o - s) % n steps from s
        # If (o - s) % n < dist_cw, the clockwise path is blocked
        blocked_cw = (o - s) % n < dist_cw
        
        # Counter-clockwise distance (decreasing index)
        # The number of steps is (s - e) % n
        dist_ccw = (s - e) % n
        # The obstacle is at (s - o) % n steps from s
        blocked_ccw = (s - o) % n < dist_ccw
        
        # It is guaranteed that the instruction is achievable, 
        # so at least one path is not blocked.
        if blocked_cw:
            return dist_ccw
        if blocked_ccw:
            return dist_cw
        return min(dist_cw, dist_ccw)

    # State: (left_pos, right_pos, total_dist)
    # Initial state: L=1, R=2, dist=0
    def process_instruction(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(process_instruction, instructions, (1, 2, 0))
    print(final_state[2])

if __name__ == "__main__":
    solve()