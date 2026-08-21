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
        # This path is valid if the block is not in the range (start, end)
        # We use modulo arithmetic to handle the wrap-around.
        
        # To simplify, we can check if the block is "between" start and end
        # in either direction.
        
        # Calculate distance going "forward" (increasing index)
        # The path is start, start+1, ..., end.
        # The number of steps is (end - start) % n.
        # This path is blocked if the block node is encountered.
        # The block is encountered if (block - start) % n < (end - start) % n.
        
        dist_fwd = (end - start) % n
        blocked_fwd = (block - start) % n < dist_fwd
        
        # Calculate distance going "backward" (decreasing index)
        # The number of steps is (start - end) % n.
        # The block is encountered if (start - block) % n < (start - end) % n.
        
        dist_bwd = (start - end) % n
        blocked_bwd = (start - block) % n < dist_bwd
        
        # It is guaranteed that the instruction is achievable.
        # If blocked_fwd is true, we must go backward, and vice versa.
        return dist_bwd if blocked_fwd else dist_fwd

    # State: (current_l, current_r, total_distance)
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
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    # Use reduce to iterate through instructions and accumulate distance
    final_state = reduce(transition, instructions, initial_state)
    
    # The result is the third element of the final state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()