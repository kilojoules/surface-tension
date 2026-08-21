import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (hand, target)
    # instructions = [('R', 4), ('L', 5), ('R', 6)]
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3])) 
        for i in range(Q)
    ]

    # Helper to calculate the shortest distance between two parts on a ring of size N
    # The distance is the minimum of the clockwise and counter-clockwise paths
    def get_dist(a, b):
        diff = abs(a - b)
        return min(diff, N - diff)

    # Helper to determine the shortest path direction and check if it's blocked
    # Returns the distance if the path is clear, otherwise returns the distance of the alternative path
    def get_actual_dist(start, end, blocker):
        # Clockwise distance (increasing index)
        # If start < end, dist is end - start. If start > end, it's (N - start) + end.
        cw_dist = (end - start) % N
        # Counter-clockwise distance (decreasing index)
        ccw_dist = (start - end) % N
        
        # Check if the blocker is on the clockwise path
        # The blocker is on the CW path if (blocker - start) % N < cw_dist
        cw_blocked = (blocker - start) % N < cw_dist
        # Check if the blocker is on the counter-clockwise path
        ccw_blocked = (start - blocker) % N < ccw_dist
        
        # We want the minimum distance that is NOT blocked.
        # Since the problem guarantees the instruction is achievable, 
        # at least one path must be clear.
        
        # If both are clear, take the minimum
        if not cw_blocked and not ccw_blocked:
            return min(cw_dist, ccw_dist)
        # If only CW is clear
        if not cw_blocked:
            return cw_dist
        # If only CCW is clear
        return ccw_dist

    # We use accumulate to track the state (left_hand, right_hand, total_dist)
    # Initial state: Left hand at 1, Right hand at 2, distance 0
    initial_state = (1, 2, 0)
    
    def transition(state, instr):
        l, r, d = state
        hand, target = instr
        if hand == 'L':
            # Move left hand to target, right hand stays at r
            move_dist = get_actual_dist(l, target, r)
            return (target, r, d + move_dist)
        else:
            # Move right hand to target, left hand stays at l
            move_dist = get_actual_dist(r, target, l)
            return (l, target, d + move_dist)

    # Process all instructions
    final_state = list(accumulate(instructions, transition, initial=initial_state))[-1]
    
    # The result is the total distance accumulated in the state
    print(final_state[2])

if __name__ == "__main__":
    solve()