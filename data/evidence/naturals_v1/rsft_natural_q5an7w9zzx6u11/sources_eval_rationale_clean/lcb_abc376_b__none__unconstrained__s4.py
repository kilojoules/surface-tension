import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between start and end on a ring of size N
    # given that a certain position 'block' cannot be entered.
    # There are two directions: clockwise and counter-clockwise.
    def get_dist(start, end, block, n):
        # Clockwise distance
        # The path is start -> start+1 -> ... -> end
        # It is blocked if 'block' is between start and end (exclusive of start, inclusive of end)
        # However, the problem says we can't move to the destination if the other hand is there.
        # But the guarantee says T_i != other_hand.
        # So we just need to check if the 'block' is encountered along the path.
        
        # Distance clockwise
        cw_dist = (end - start) % n
        # The block is encountered if (block - start) % n < cw_dist
        # Wait, the rule is: cannot move to destination if other hand is there.
        # Since we move one step at a time, we are blocked if the block is anywhere 
        # on the path from start to end.
        
        # A simpler way to check if a path is blocked:
        # In a ring, there are only two paths. 
        # Path 1: start -> start+1 -> ... -> end
        # Path 2: start -> start-1 -> ... -> end
        # Path 1 is blocked if 'block' is reached.
        # Path 2 is blocked if 'block' is reached.
        
        # Check clockwise:
        # The nodes visited are (start + k) % n for k = 1 to cw_dist
        # It's blocked if (block - start) % n is between 1 and cw_dist
        is_cw_blocked = 0 < (block - start) % n <= cw_dist
        
        # Check counter-clockwise:
        ccw_dist = (start - end) % n
        # The nodes visited are (start - k) % n for k = 1 to ccw_dist
        # It's blocked if (start - block) % n is between 1 and ccw_dist
        is_ccw_blocked = 0 < (start - block) % n <= ccw_dist
        
        # Return the minimum of the unblocked paths. 
        # Use a large number for blocked paths.
        return min(
            cw_dist if not is_cw_blocked else float('inf'),
            ccw_dist if not is_ccw_blocked else float('inf')
        )

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    # Note: The problem uses 1-indexing, we convert to 0-indexing for easier modulo
    initial_state = (0, 1, 0)
    
    def own_reduce(state, instr):
        l, r, total = state
        h, t = instr
        t -= 1 # 0-indexed
        
        if h == 'L':
            # Move left hand to t, right hand r is blocked
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l is blocked
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(own_reduce, instructions, initial_state)
    print(int(final_state[2]))

if __name__ == "__main__":
    solve()