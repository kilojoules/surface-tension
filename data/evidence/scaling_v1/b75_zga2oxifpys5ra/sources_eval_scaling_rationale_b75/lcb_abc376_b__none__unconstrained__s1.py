import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T)
    instructions = []
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        instructions.append((h, t))

    def get_dist(start, end, blocked, n):
        # Calculate distance from start to end on a ring of size n
        # without passing through the 'blocked' node.
        # There are two directions: clockwise and counter-clockwise.
        
        # Clockwise distance
        dist_cw = (end - start) % n
        # Check if blocked node is in the clockwise path
        # The path is start -> start+1 -> ... -> end
        # Blocked is in path if (blocked - start) % n < dist_cw
        # However, the problem says we can't move to the destination if the other hand is there.
        # The constraint is: we cannot step on the 'blocked' node.
        
        # A simpler way to check if the blocked node is on the path:
        # In a ring of N, there are only two paths between any two points.
        # One path is length d, the other is N - d.
        # One of these paths must contain the blocked node unless the blocked node
        # is one of the endpoints (but the problem guarantees T != blocked).
        
        # Let's find the distance of the path that does NOT contain the blocked node.
        # Path 1: start -> (start+1)%N -> ... -> end
        # Path 2: start -> (start-1)%N -> ... -> end
        
        # Check if blocked is "between" start and end clockwise
        # Normalize everything to 0..N-1 for easier modulo math
        s, e, b = (start-1)%n, (end-1)%n, (blocked-1)%n
        
        # Clockwise distance
        d_cw = (e - s) % n
        # Blocked is on clockwise path if (b - s) % n < d_cw
        is_blocked_cw = (b - s) % n < d_cw
        
        # Counter-clockwise distance
        d_ccw = (s - e) % n
        # Blocked is on counter-clockwise path if (s - b) % n < d_ccw
        is_blocked_ccw = (s - b) % n < d_ccw
        
        # Since it's guaranteed the move is achievable, one path must be clear.
        if is_blocked_cw:
            return d_ccw
        if is_blocked_ccw:
            return d_cw
        # If neither is blocked (only possible if N=2, but N>=3), return min
        return min(d_cw, d_ccw)

    # State: (left_pos, right_pos, total_dist)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)
    
    def transition(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r is blocked
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l is blocked
            d = get으로_dist(r, t, l, N) # Wait, typo in function name
            return (l, t, total + d)

    # Correcting the transition function to use the defined get_dist
    def transition_fixed(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            return (t, r, total + get_dist(l, t, r, N))
        else:
            return (l, t, total + get_dist(r, t, l, N))

    final_state = reduce(transition_fixed, instructions, initial_state)
    print(final_state[2])

# To avoid the function call overhead and keep it clean, 
# I will wrap the logic inside a block and use the helper.
if __name__ == "__main__":
    # Redefining the logic slightly to ensure it fits the "no loop" constraint strictly
    # by using map/reduce/comprehensions.
    import sys
    from functools import reduce

    def solve_final():
        input_data = sys.stdin.read().split()
        if not input_data: return
        N = int(input_data[0])
        Q = int(input_data[1])
        
        # Use a list comprehension to parse instructions
        instructions = [(input_data[i], int(input_data[i+1])) 
                        for i in range(2, len(input_data), 2)]

        def get_dist(s_raw, e_raw, b_raw, n):
            s, e, b = (s_raw-1)%n, (e_raw-1)%n, (b_raw-1)%n
            d_cw = (e - s) % n
            is_blocked_cw = (b - s) % n < d_cw
            d_ccw = (s - e) % n
            is_blocked_ccw = (s - b) % n < d_ccw
            return d_ccw if is_blocked_cw else d_cw

        def transition(state, instr):
            l, r, total = state
            h, t = instr
            return (t, r, total + get_dist(l, t, r, N)) if h == 'L' else \
                   (l, t, total + get_dist(r, t, l, N))

        print(reduce(transition, instructions, (1, 2, 0))[2])

    solve_final()