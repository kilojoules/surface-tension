import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    # Using list comprehension to avoid for-loops
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3])) 
        for i in range(Q)
    ]
    
    # Function to calculate the shortest distance between two points on a ring
    # given that a certain point (the other hand) is blocking the path.
    # The distance is the minimum of clockwise and counter-clockwise paths,
    # provided the blocking point is not on that path.
    def get_dist(start, end, block, n):
        # Clockwise distance (increasing index)
        # Path: start -> start+1 -> ... -> end
        # The block is on the path if block is between start and end (exclusive)
        # Handle wrap-around by normalizing coordinates
        
        # We can simplify: there are only two directions.
        # Direction 1: (start + k) % N
        # Direction 2: (start - k) % N
        
        # To avoid loops, we check if the 'block' is in the way.
        # A path from s to e is blocked if 'block' lies on the arc.
        
        # Normalize to 0-indexed
        s, e, b = start - 1, end - 1, block - 1
        
        # Clockwise distance
        dist_cw = (e - s) % n
        # The block is clockwise from s if (b - s) % n < dist_cw
        # However, the problem says we can't move to the destination if the other hand is there.
        # But the guarantee says T_i != other_hand.
        # The block is "in the way" if it's any of the steps between s and e.
        blocked_cw = (b - s) % n < dist_cw and b != s
        
        # Counter-clockwise distance
        dist_ccw = (s - e) % n
        blocked_ccw = (s - b) % n < dist_ccw and b != s
        
        # Return the minimum of the unblocked paths.
        # Since it's guaranteed to be achievable, at least one is unblocked.
        # We use a large number for blocked paths.
        return min(
            dist_cw if not blocked_cw else float('inf'),
            dist_ccw if not blocked_ccw else float('inf')
        )

    # State: (left_pos, right_pos, total_distance)
    # Initial state: Left=1, Right=2, Dist=0
    initial_state = (1, 2, 0)
    
    def transition(state, instr):
        l, r, d = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r stays
            move_dist = get_dist(l, t, r, N)
            return (t, r, d + move_dist)
        else:
            # Move right hand to t, left hand l stays
            move_dist = get_dist(r, t, l, N)
            return (l, t, d + move_dist)

    # Use reduce to simulate the sequence of instructions
    final_state = reduce(transition, instructions, initial_state)
    
    # The result is the accumulated distance
    print(int(final_state[2]))

if __name__ == "__main__":
    solve()