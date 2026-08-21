import sys
from itertools import accumulate

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (hand, target)
    # H_i is at index 2 + 2*i, T_i is at index 3 + 2*i
    instructions = [
        (input_data[2 + 2*i], int(input_data[3 + 2*i]))
        for i in range(Q)
    ]

    # Helper to calculate shortest distance between two points on a ring of size N
    # without passing through a forbidden point.
    # Since we can only move one hand, the other hand acts as a wall.
    # The distance is simply the length of the path that doesn't contain the wall.
    def get_dist(start, end, wall):
        if start == end:
            return 0
        
        # There are two paths on the ring: clockwise and counter-clockwise.
        # One path is blocked by the 'wall'.
        # We need to find the length of the path that is NOT blocked.
        
        # Normalize coordinates to 0...N-1 for easier modulo arithmetic
        s, e, w = start - 1, end - 1, wall - 1
        
        # Path 1: s -> (s+1)%N -> ... -> e
        # Length is (e - s) % N
        # This path is blocked if the wall is any of the steps between s and e.
        # The wall is on this path if (w - s) % N < (e - s) % N.
        
        dist_cw = (e - s) % N
        is_blocked_cw = (w - s) % N < dist_cw
        
        # Path 2: s -> (s-1)%N -> ... -> e
        # Length is (s - e) % N
        # This path is blocked if (s - w) % N < (s - e) % N.
        
        dist_ccw = (s - e) % N
        is_blocked_ccw = (s - w) % N < dist_ccw
        
        # The problem guarantees the instruction is achievable.
        # We return the distance of the path that is not blocked.
        return dist_cw if not is_blocked_cw else dist_ccw

    # We use accumulate to track the state (left_hand, right_hand, total_dist)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)
    
    def transition(state, instr):
        l, r, d = state
        hand, target = instr
        if hand == 'L':
            # Move left hand to target, right hand is the wall
            cost = get_dist(l, target, r)
            return (target, r, d + cost)
        else:
            # Move right hand to target, left hand is the wall
            cost = get_dist(r, target, l)
            return (l, target, d + cost)

    # Process all instructions
    final_state = list(accumulate(instructions, transition, initial=initial_state))[-1]
    
    # The result is the total distance accumulated in the state
    print(final_state[2])

if __name__ == "__main__":
    solve()