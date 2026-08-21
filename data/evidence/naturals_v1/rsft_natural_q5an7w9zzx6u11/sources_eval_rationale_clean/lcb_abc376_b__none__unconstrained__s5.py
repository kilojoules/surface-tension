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

    # Helper to calculate distance between two points on a ring of size N
    # given that a certain point 'obstacle' cannot be passed.
    # Since we can't use loops, we evaluate both directions (clockwise/counter-clockwise)
    # and check if the obstacle lies on that path.
    def get_dist(start, end, obstacle, n):
        # Clockwise distance
        cw_dist = (end - start) % n
        # The obstacle is at 'obstacle'. We check if it's between start and end clockwise.
        # A point p is on the clockwise path from s to e if (p-s)%n < (e-s)%n.
        # However, the obstacle is only a problem if it's strictly between them.
        # But the problem says we can't move to the destination if the other hand is there.
        # The guarantee says T_i != obstacle, so we just need to check if the path is blocked.
        
        # Path 1: start -> start+1 -> ... -> end
        # Blocked if (obstacle - start) % n < cw_dist
        blocked_cw = (obstacle - start) % n < cw_dist
        
        # Path 2: start -> start-1 -> ... -> end
        # CCW distance
        ccw_dist = (start - end) % n
        # Blocked if (start - obstacle) % n < ccw_dist
        blocked_ccw = (start - obstacle) % n < ccw_dist
        
        # We return the distance of the unblocked path. 
        # It is guaranteed that at least one path is always open.
        return cw_dist if not blocked_cw else ccw_dist

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    def transition(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r is obstacle
            # Adjust 1-based indexing to 0-based for modulo math
            d = get_dist(l - 1, t - 1, r - 1, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l is obstacle
            d = get_dist(r - 1, t - 1, l - 1, N)
            return (l, t, total + d)

    # Use reduce to simulate the process across all instructions
    final_state = reduce(transition, instructions, (1, 2, 0))
    
    # The result is the accumulated distance
    print(final_state[2])

if __name__ == "__main__":
    solve()