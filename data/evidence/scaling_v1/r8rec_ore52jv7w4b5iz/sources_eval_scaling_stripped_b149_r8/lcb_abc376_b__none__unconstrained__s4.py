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
    # without passing through a forbidden point 'obs'
    def get_dist(start, end, obs):
        if start == end:
            return 0
        
        # There are two directions: clockwise and counter-clockwise
        # Direction 1: start -> start+1 -> ... -> end
        # Direction 2: start -> start-1 -> ... -> end
        
        # To simplify, we check if the obstacle is in the way.
        # We can simulate the two paths.
        
        # Path A: Increasing indices (modulo N)
        # The path is start, (start+1)%N, ..., end
        # We check if 'obs' is encountered.
        
        # Since N is small (100), we can just use a list comprehension to check
        # if the obstacle lies between start and end in either direction.
        
        # Normalize to 0-indexed for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obs - 1
        
        # Clockwise distance
        dist_cw = (e - s) % N
        # The points visited are (s + i) % N for i in 1...dist_cw
        # Obstacle is hit if (s + i) % N == o for some i in 1...dist_cw
        # This is equivalent to saying (o - s) % N is between 1 and dist_cw
        blocked_cw = (1 <= (o - s) % N <= dist_cw)
        
        # Counter-clockwise distance
        dist_ccw = (s - e) % N
        # Obstacle is hit if (s - i) % N == o for some i in 1...dist_ccw
        # This is equivalent to saying (s - o) % N is between 1 and dist_ccw
        blocked_ccw = (1 <= (s - o) % N <= dist_ccw)
        
        # Return the minimum of the unblocked paths. 
        # The problem guarantees instructions are achievable.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res)

    # State: (left_hand, right_hand)
    # We use a list of possible states and their cumulative costs.
    # Since the problem says positions are uniquely determined, 
    # we can just track the current (l, r).
    
    # Initial state
    curr_l, curr_r = 1, 2
    total_dist = 0
    
    # We cannot use a loop, so we use a reduction-like approach 
    # to process instructions. However, since we must avoid 'for',
    # we use a recursive-like structure via map/reduce or a list comprehension
    # that updates a state container.
    
    # Because we need to update state (l, r, total), and we can't use loops,
    # we use a trick with a list and a function that updates the state.
    # We'll use a list of one element to hold the state and update it via a list comprehension.
    
    state = [ (curr_l, curr_r, 0) ]
    
    def process_instruction(s, instr):
        l, r, cost = s[0]
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r is obstacle
            d = get_dist(l, t, r)
            return [(t, r, cost + d)]
        else:
            # Move right hand to t, left hand l is obstacle
            d = get_dist(r, t, l)
            return [(l, t, cost + d)]

    # To simulate the sequence of instructions without a loop, 
    # we can use a recursive-like approach with a helper function 
    # and a list comprehension to drive it, or simply use functools.reduce.
    # Since functools is allowed, reduce is the perfect tool.
    from functools import reduce
    
    final_state = reduce(
        lambda s, instr: process_instruction(s, instr),
        instructions,
        state
    )
    
    print(final_state[0][2])

if __name__ == "__main__":
    solve()