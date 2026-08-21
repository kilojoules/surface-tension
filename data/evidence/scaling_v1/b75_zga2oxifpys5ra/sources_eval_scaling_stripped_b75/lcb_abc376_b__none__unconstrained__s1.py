import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions
    # instructions will be a list of (H_i, T_i)
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]

    # Initial state: Left hand at 1, Right hand at 2
    # State is (L, R)
    # We use a list of possible states after each instruction.
    # Since the problem guarantees positions are uniquely determined,
    # we can just track the current (L, R).
    
    def get_dist(start, end, obstacle, n):
        """
        Calculate min distance from start to end on a ring of size n,
        given that the 'obstacle' position cannot be stepped on.
        Returns infinity if unreachable.
        """
        # There are two directions: clockwise and counter-clockwise.
        # Clockwise: start -> start+1 -> ... -> end
        # Counter-clockwise: start -> start-1 -> ... -> end
        
        # Normalize to 0-indexed for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Path 1: Increasing index (s -> s+1 -> ... -> e)
        # The distance is (e - s) % n. 
        # The path is blocked if the obstacle is in the range (s, e) modulo n.
        dist_cw = (e - s) % n
        # Check if obstacle is between s and e clockwise
        # Obstacle is at relative position (o - s) % n
        blocked_cw = (0 < (o - s) % n < dist_cw)
        
        # Path 2: Decreasing index (s -> s-1 -> ... -> e)
        # The distance is (s - e) % n.
        dist_ccw = (s - e) % n
        # Check if obstacle is between s and e counter-clockwise
        blocked_ccw = (0 < (s - o) % n < dist_ccw)
        
        res = float('inf')
        if not blocked_cw: res = min(res, dist_cw)
        if not blocked_ccw: res = min(res, dist_ccw)
        return res

    # We use a list comprehension to simulate the state transitions.
    # Since we need the total distance, we carry (current_L, current_R, total_dist).
    # We use a helper function and reduce-like logic via a loop.
    
    def transition(state, instr):
        curr_l, curr_r, total_d = state
        h, t = instr
        if h == 'L':
            d = get_dist(curr_l, t, curr_r, N)
            return (t, curr_r, total_d + d)
        else:
            d = get_dist(curr_r, t, curr_l, N)
            return (curr_l, t, total_d + d)

    # Use a loop to process all instructions
    final_state = state = (1, 2, 0)
    for instr in instructions:
        state = transition(state, instr)
    
    print(state[2])

if __name__ == "__main__":
    solve()