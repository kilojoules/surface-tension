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
    instructions = []
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        instructions.append((h, t))

    # Helper to calculate distance between two points on a ring of size N
    # given that a certain point 'obstacle' cannot be passed.
    # The distance is the number of edges.
    def get_dist(start, end, obstacle, n):
        if start == end:
            return 0
        
        # There are two directions to move on a ring: clockwise and counter-clockwise.
        # One direction is blocked by the obstacle.
        # We need to find the distance of the path that does NOT contain the obstacle.
        
        # Normalize coordinates to 0...N-1 for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Distance moving "forward" (s -> s+1 -> ...)
        # The forward path is blocked if the obstacle is between s and e (modulo N)
        # A point 'o' is between 's' and 'e' clockwise if:
        # if s < e: s < o < e
        # if s > e: o > s or o < e
        is_blocked_forward = (s < o < e) if s < e else (o > s or o < e)
        
        # Distance moving "backward" (s -> s-1 -> ...)
        # The backward path is blocked if the obstacle is between s and e (modulo N) counter-clockwise
        # Which is the same as saying the forward path is NOT blocked.
        
        dist_forward = (e - s) % n
        dist_backward = (s - e) % n
        
        return dist_backward if is_blocked_forward else dist_forward

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is obstacle
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l is obstacle
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()