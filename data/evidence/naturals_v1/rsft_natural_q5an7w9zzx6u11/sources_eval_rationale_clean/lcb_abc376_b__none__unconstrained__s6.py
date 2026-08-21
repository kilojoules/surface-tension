import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as a list of (H, T) tuples
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]

    # Helper to calculate the shortest distance between two points on a ring of size N
    # given that the other hand (obstacle) is at position 'obs'.
    # On a ring, there are two directions. One might be blocked by the obstacle.
    def get_dist(start, end, obs, n):
        # Clockwise distance (increasing index)
        # The path is start -> start+1 -> ... -> end
        # It is blocked if obs is between start and end (exclusive of start, inclusive of end)
        # However, the problem says we can't move to the part the other hand is holding.
        # So if obs is anywhere on the path, that path is blocked.
        
        # Distance clockwise
        d_cw = (end - start) % n
        # Check if obs is on the clockwise path:
        # The parts visited are (start + k) % n for k = 1...d_cw
        # Obs is on path if (obs - start) % n is between 1 and d_cw
        blocked_cw = (0 < (obs - start) % n <= d_cw)
        
        # Distance counter-clockwise
        d_ccw = (start - end) % n
        # Obs is on path if (start - obs) % n is between 1 and d_ccw
        blocked_ccw = (0 < (start - obs) % n <= d_ccw)
        
        # We need to return the minimum of the unblocked paths.
        # It is guaranteed that at least one path is always open.
        return min([d for d, blocked in [(d_cw, blocked_cw), (d_ccw, blocked_ccw)] if not blocked])

    # State: (left_pos, right_pos, total_dist)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r stays put
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l stays put
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    # Use reduce to process all instructions and accumulate the result
    final_state = reduce(transition, instructions, initial_state)
    
    # The result is the third element of the final state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()