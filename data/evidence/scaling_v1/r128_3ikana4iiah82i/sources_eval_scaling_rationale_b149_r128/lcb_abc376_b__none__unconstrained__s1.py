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
    # clockwise distance from a to b
    def get_dist(a, b):
        return (b - a) % N if (b - a) % N != 0 else 0

    # The core logic to calculate the distance the moving hand must travel
    # given the position of the stationary hand.
    # On a ring, there are two paths: clockwise and counter-clockwise.
    # One path is blocked if the stationary hand is on it.
    def calc_move(current, target, obstacle):
        # Distance clockwise from current to target
        d_cw = (target - current) % N
        # Distance counter-clockwise from current to target
        d_ccw = (current - target) % N
        
        # The stationary hand (obstacle) blocks a path if it lies between 
        # current and target.
        # Check if obstacle is in the clockwise path:
        # The clockwise path consists of points (current + k) % N
        # The obstacle is in the way if (obstacle - current) % N < (target - current) % N
        # Note: current and target are 1-indexed, so we normalize to 0-indexed for modulo.
        
        c, t, o = current - 1, target - 1, obstacle - 1
        
        # Clockwise distance and check if obstacle is encountered
        # The clockwise path is blocked if the distance from c to o is less than c to t.
        dist_c_o = (o - c) % N
        dist_c_t = (t - c) % N
        
        # If dist_c_o < dist_c_t, the clockwise path is blocked.
        # Otherwise, the counter-clockwise path is blocked (or neither if N=2, but N>=3).
        # Actually, the only way a path is NOT blocked is if the obstacle is not 
        # in the way.
        # Since we can't pass the other hand, we must take the path that doesn't contain 'o'.
        
        # Path 1: Clockwise. Blocked if (o-c)%N < (t-c)%N
        # Path 2: Counter-clockwise. Blocked if (c-o)%N < (c-t)%N
        
        # Because it's guaranteed that the instruction is achievable,
        # one of the two paths must be clear.
        
        # If the clockwise path is blocked, we must take the counter-clockwise path.
        # If the counter-clockwise path is blocked, we must take the clockwise path.
        # If neither is blocked (not possible here as the hand is always "somewhere"),
        # we take the minimum. However, the obstacle is always at some position.
        
        # A path is blocked if the obstacle lies strictly between the start and end.
        # Clockwise blocked if: 0 < (o-c)%N < (t-c)%N
        # Counter-clockwise blocked if: 0 < (c-o)%N < (c-t)%N
        
        is_cw_blocked = 0 < (o - c) % N < (t - c) % N
        is_ccw_blocked = 0 < (c - o) % N < (c - t) % N
        
        # We want the minimum distance of the non-blocked path.
        # If both are open (only possible if o is at c or t, but problem says o != t),
        # we take min(d_cw, d_ccw).
        
        res = [
            (t - c) % N if not is_cw_blocked else float('inf'),
            (c - t) % N if not is_ccw_blocked else float('inf')
        ]
        return min(res)

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)
    
    # Use reduce to iterate through instructions and accumulate distance
    final_state = reduce(
        lambda state, instr: (
            # New Left Hand
            (instr[1], state[1], state[2] + calc_move(state[0], instr[1], state[1]))
            if instr[0] == 'L' else
            # New Right Hand
            (state[0], instr[1], state[2] + calc_move(state[1], instr[1], state[0]))
        ),
        instructions,
        initial_state
    )
    
    print(final_state[2])

if __name__ == "__main__":
    solve()