import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as pairs of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance on a ring avoiding a specific point
    # The ring is 1-indexed.
    # To move from 'start' to 'end' avoiding 'obs':
    # There are two directions: clockwise and counter-clockwise.
    # One direction is blocked if 'obs' lies on the path.
    def get_dist(start, end, obs, n):
        if start == end:
            return 0
        
        # Clockwise distance (start -> start+1 -> ...)
        # The number of steps is (end - start) mod n
        cw_dist = (end - start) % n
        # The path is blocked if obs is between start and end clockwise
        # obs is on cw path if (obs - start) % n < cw_dist
        cw_blocked = (obs - start) % n < cw_dist
        
        # Counter-clockwise distance (start -> start-1 -> ...)
        # The number of steps is (start - end) mod n
        ccw_dist = (start - end) % n
        # obs is on ccw path if (start - obs) % n < ccw_dist
        ccw_blocked = (start - obs) % n < ccw_dist
        
        # We are guaranteed the instruction is achievable.
        # Return the distance of the non-blocked path.
        # If both are open (not possible here since obs is one point), min would apply.
        # But since we can't jump over the other hand, only one direction is viable
        # unless the other hand is not in the way.
        # Actually, the only way both are open is if N=2, but N >= 3.
        # With N >= 3, the other hand always blocks one of the two arcs.
        return cw_dist if not cw_blocked else ccw_dist

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r stays
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l stays
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    # Use reduce to simulate the sequence of instructions
    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()