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
    # We use a list comprehension to parse the remaining input
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # The state is (current_l, current_r, total_distance)
    # Initial state: Left hand at 1, Right hand at 2, distance 0
    initial_state = (1, 2, 0)

    def get_dist(start, end, obstacle, n):
        # To move from start to end on a ring of size N without passing through 'obstacle',
        # there are two possible directions: clockwise and counter-clockwise.
        # However, the problem states we cannot move to the part the other hand is holding.
        # This means the obstacle blocks one of the two paths.
        
        # Normalize to 0-indexed for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Path 1: Increasing index (s -> s+1 -> ... -> e)
        # Path 2: Decreasing index (s -> s-1 -> ... -> e)
        
        # Check if obstacle is in the path from s to e moving clockwise
        # Clockwise distance
        cw_dist = (e - s) % n
        # The obstacle is encountered if (o - s) % n < cw_dist
        cw_blocked = (o - s) % n < cw_dist and o != s
        
        # Counter-clockwise distance
        ccw_dist = (s - e) % n
        # The obstacle is encountered if (s - o) % n < ccw_dist
        ccw_blocked = (s - o) % n < ccw_dist and o != s
        
        # Since the problem guarantees the instruction is achievable,
        # one of these paths must be clear.
        if not cw_blocked:
            # We can't simply take min() because the obstacle might block one way.
            # But we want the minimum distance of the UNBLOCKED path.
            # If both are unblocked, take the minimum.
            if not ccw_blocked:
                return min(cw_dist, ccw_dist)
            return cw_dist
        if not ccw_blocked:
            return ccw_dist
        
        # This part should not be reached given the problem guarantees
        return float('inf')

    def process_instruction(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is obstacle
            dist = get_dist(l, t, r, N)
            return (t, r, total + dist)
        else:
            # Move right hand to t, left hand l is obstacle
            dist = get_dist(r, t, l, N)
            return (l, t, total + dist)

    # Use reduce to accumulate the state over all instructions
    final_state = reduce(process_instruction, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()