import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H_i, T_i)
    instructions = []
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        instructions.append((h, t))

    # Helper to calculate distance between two points on a ring of size N
    # given that a certain point 'block' cannot be passed.
    # The hands cannot jump over each other.
    # There are two directions: clockwise and counter-clockwise.
    # One direction will be blocked by the other hand.
    def get_dist(start, end, block, n):
        if start == end:
            return 0
        
        # We need to find the distance from start to end without passing through 'block'.
        # On a ring, there are only two paths. One must be blocked.
        # Let's normalize coordinates to 0...N-1
        s, e, b = start - 1, end - 1, block - 1
        
        # Path 1: Increasing index (s -> s+1 -> ... -> e)
        # This path is clear if 'b' is not between s and e (modulo N)
        # To simplify, we can check if the path is blocked by checking 
        # if b is "between" s and e in the clockwise direction.
        
        # Clockwise distance
        cw_dist = (e - s + n) % n
        # Is block b on the clockwise path from s to e?
        # b is on the path if (b - s + n) % n < cw_dist
        is_cw_blocked = (b - s + n) % n < cw_dist and s != b
        
        # Counter-clockwise distance
        ccw_dist = (s - e + n) % n
        # Is block b on the counter-clockwise path from s to e?
        is_ccw_blocked = (s - b + n) % n < ccw_dist and s != b
        
        # The problem guarantees the instruction is achievable.
        # Only one path will be available because the other hand blocks the way.
        if not is_cw_blocked:
            return cw_dist
        if not is_ccw_blocked:
            return ccw_dist
        
        # This part should theoretically not be reached given problem constraints
        return float('inf')

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total_d = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is stationary
            d = get_dist(l, t, r, N)
            return (t, r, total_d + d)
        else:
            # Move right hand to t, left hand l is stationary
            d = get_dist(r, t, l, N)
            return (l, t, total_d + d)

    # Use reduce to process all instructions
    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()