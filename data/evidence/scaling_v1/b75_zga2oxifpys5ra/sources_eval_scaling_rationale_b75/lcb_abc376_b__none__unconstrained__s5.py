import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Function to calculate the shortest distance between start and end on a ring of size N,
    # given that the other hand is at position 'obstacle'.
    # The movement must not pass through the obstacle.
    def get_dist(start, end, obstacle, n):
        # There are two directions to move on a ring: clockwise and counter-clockwise.
        # We check if the obstacle blocks the path in either direction.
        
        # Helper to check if obstacle is between a and b moving clockwise
        # On a ring 1...N, moving clockwise from a to b:
        # If a < b, range is [a, b]. If a > b, range is [a, N] union [1, b].
        def is_blocked(s, e, obs, n):
            if s <= e:
                return s < obs < e
            else:
                return obs > s or obs < e

        # Clockwise distance
        cw_dist = (end - start) % n
        # Counter-clockwise distance
        ccw_dist = (start - end) % n
        
        # We can move clockwise if the obstacle is not in the path.
        # Note: the problem says we cannot move to the destination if the other hand is there.
        # It is guaranteed T_i != obstacle.
        # The path is blocked if the obstacle is strictly between start and end.
        
        # Check clockwise path
        # Clockwise path visits: start, start+1, ..., end
        # It is blocked if obstacle is one of the intermediate steps.
        blocked_cw = is_blocked(start, end, obstacle, n)
        blocked_ccw = is_blocked(end, start, obstacle, n)
        
        # We want the minimum distance among non-blocked paths.
        # Since it's guaranteed achievable, at least one path is open.
        res = []
        if not blocked_cw: res.append(cw_dist)
        if not blocked_ccw: res.append(ccw_dist)
        return min(res)

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)
    
    # Use reduce to process instructions and accumulate distance
    final_state = reduce(
        lambda state, instr: (
            # New state tuple: (new_l, new_r, new_dist)
            (
                instr[1] if instr[0] == 'L' else state[0], # new_l
                instr[1] if instr[0] == 'R' else state[1], # new_r
                state[2] + (
                    get_dist(state[0], instr[1], state[1], N) 
                    if instr[0] == 'L' 
                    else get_dist(state[1], instr[1], state[0], N)
                ) # new_dist
            )
        ),
        instructions,
        initial_state
    )
    
    print(final_state[2])

if __name__ == "__main__":
    solve()