import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    # input_data[2:] contains H1, T1, H2, T2, ...
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    def get_dist(start, end, blocker, n):
        # There are two paths on a ring: clockwise and counter-clockwise.
        # Path 1: start -> start+1 -> ... -> end (modulo n)
        # Path 2: start -> start-1 -> ... -> end (modulo n)
        
        # Normalize to 0-indexed
        s, e, b = start - 1, end - 1, blocker - 1
        
        # Clockwise distance
        cw_dist = (e - s) % n
        # Check if blocker is on the clockwise path
        # Blocker is on path if (blocker - s) % n < cw_dist
        cw_blocked = (b - s) % n < cw_dist and s != b
        
        # Counter-clockwise distance
        ccw_dist = (s - e) % n
        # Blocker is on path if (s - blocker) % n < ccw_dist
        ccw_blocked = (s - b) % n < ccw_dist and s != b
        
        # We are guaranteed the move is possible.
        # If one path is blocked, we must take the other.
        if cw_blocked:
            return ccw_dist
        if ccw_blocked:
            return cw_dist
        # If neither is blocked (only possible if N is large enough), take minimum
        return min(cw_dist, ccw_dist)

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def process_instruction(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r stays put
            dist = get_dist(l, t, r, N)
            return (t, r, total + dist)
        else:
            # Move right hand to t, left hand l stays put
            dist = get_dist(r, t, l, N)
            return (l, t, total + dist)

    final_state = reduce(process_instruction, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()