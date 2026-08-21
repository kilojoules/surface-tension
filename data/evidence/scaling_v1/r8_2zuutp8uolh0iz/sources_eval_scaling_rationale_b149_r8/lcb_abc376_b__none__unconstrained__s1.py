import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between two points on a ring of size N
    # given that the other hand (blocker) is at position 'b'.
    # The hand cannot pass through 'b'.
    def get_dist(start, end, blocker, n):
        # There are two arcs between start and end.
        # One arc is (start -> start+1 -> ... -> end)
        # The other is (start -> start-1 -> ... -> end)
        
        # Clockwise distance
        dist_cw = (end - start) % n
        # Counter-clockwise distance
        dist_ccw = (start - end) % n
        
        # Check if blocker is on the clockwise path
        # Blocker is on clockwise path if (blocker - start) % n < dist_cw
        # and blocker != start.
        blocked_cw = (blocker != start) and ((blocker - start) % n < dist_cw)
        
        # Check if blocker is on the counter-clockwise path
        blocked_ccw = (blocker != start) and ((start - blocker) % n < dist_ccw)
        
        # Since the problem guarantees the move is possible, 
        # at least one path must be clear.
        if blocked_cw:
            return dist_ccw
        if blocked_ccw:
            return dist_cw
        # If neither is blocked, take the minimum
        return min(dist_cw, dist_ccw)

    # State: (left_pos, right_pos, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand from l to t, right hand r is blocker
            # Using 0-indexed logic for modulo, so we use (pos-1)
            d = get_dist(l - 1, t - 1, r - 1, N)
            return (t, r, total + d)
        else:
            # Move right hand from r to t, left hand l is blocker
            d = get_dist(r - 1, t - 1, l - 1, N)
            return (l, t, total + d)

    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()