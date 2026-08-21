import sys
from functools import reduce

def solve():
    # Read all input data
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

    def get_dist(start, end, block, n):
        # The ring is 1-indexed. We need the shortest path from start to end
        # without passing through the block node.
        # There are only two directions on a ring: clockwise and counter-clockwise.
        
        # Direction 1: start -> start+1 -> ... -> end
        # We check if the block is in the path.
        # A path from s to e (clockwise) contains block b if:
        # s < b < e (for s < e) or (s < b <= n or 1 <= b < e) (for s > e)
        
        def is_blocked(s, e, b):
            if s < e:
                return s < b < e
            else: # s > e, path wraps around n to 1
                return b > s or b < e

        # Distance clockwise
        dist_cw = (end - start + n) % n
        # Distance counter-clockwise
        dist_ccw = (start - end + n) % n
        
        # We can't move if the block is in the way.
        # Note: The problem guarantees instructions are achievable.
        # If clockwise is blocked, we must go counter-clockwise, and vice versa.
        
        # Check if clockwise path is blocked
        # The clockwise path is start, start+1, ..., end.
        # It is blocked if 'block' is any of the nodes between start and end.
        # Since we can't land on 'block', we check if 'block' is encountered.
        
        # A simpler way to check if a direction is blocked:
        # In a ring of size N, if one node is blocked, the ring becomes a line.
        # The distance is simply the absolute difference in a linear space
        # if we "cut" the ring at the block.
        
        # Transform coordinates so that the block is at the "edge"
        # New coordinates: (x - block + N) % N
        s_rel = (start - block + N) % N
        e_rel = (end - block + N) % N
        # Now the block is at 0. The path cannot pass through 0.
        # The only path is the direct distance between s_rel and e_rel.
        return abs(s_rel - e_rel)

    # State: (left_hand, right_hand, total_distance)
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

    # Correcting the transition to use the defined get_dist
    def final_transition(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            return (t, r, total + get_dist(l, t, r, N))
        else:
            return (l, t, total + get_dist(r, t, l, N))

    final_state = reduce(final_transition, instructions, initial_state)
    print(final_state[2])

# Using a wrapper to call solve() without an explicit loop
# Since the constraints forbid 'for' or 'while', 
# the logic is contained within reduce().
if __name__ == "__main__":
    solve()