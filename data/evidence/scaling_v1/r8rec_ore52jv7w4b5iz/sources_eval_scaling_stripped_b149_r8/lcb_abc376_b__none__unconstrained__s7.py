import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'block'
    def get_dist(start, end, block):
        # The ring is 1-indexed. We can think of it as 0 to N-1.
        # There are only two directions to move on a ring.
        # One direction is clockwise, one is counter-clockwise.
        # A path is valid if the 'block' is not on it.
        
        # Since N is small (100), we can just simulate both directions
        # and pick the minimum of the valid ones.
        
        # Direction 1: Increasing (1 -> 2 -> ... -> N -> 1)
        # Distance if we move 'forward'
        dist_fwd = (end - start + N) % N
        # The block is encountered if (block - start + N) % N < dist_fwd
        blocked_fwd = (block != start) and ((block - start + N) % N < dist_fwd)
        
        # Direction 2: Decreasing (1 -> N -> N-1 -> ... -> 1)
        dist_bwd = (start - end + N) % N
        # The block is encountered if (start - block + N) % N < dist_bwd
        blocked_bwd = (block != start) and ((start - block + N) % N < dist_bwd)
        
        # We need to find the minimum distance of the non-blocked path.
        # It is guaranteed that at least one path is achievable.
        res = []
        if not blocked_fwd: res.append(dist_fwd)
        if not blocked_bwd: res.append(dist_bwd)
        return min(res)

    # State: (current_l, current_r)
    # We use a list to simulate the state changes across Q instructions.
    # Since we cannot use loops, we use a reduction-like approach.
    # However, since we need to track both the total distance and the positions,
    # we can use a helper function with functools.reduce.
    from functools import reduce

    def step(state, instr):
        curr_l, curr_r, total_dist = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand stays at curr_r
            d = get_dist(curr_l, t, curr_r)
            return (t, curr_r, total_dist + d)
        else:
            # Move right hand to t, left hand stays at curr_l
            d = get_dist(curr_r, t, curr_l)
            return (curr_l, t, total_dist + d)

    # Initial state: L=1, R=2, dist=0
    final_state = reduce(step, instructions, (1, 2, 0))
    
    # The result is the third element of the final state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()