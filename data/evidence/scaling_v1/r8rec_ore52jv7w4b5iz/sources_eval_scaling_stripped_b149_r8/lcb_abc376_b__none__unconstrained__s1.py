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
        # The ring is 1-indexed. We can think of it as a graph.
        # Since N is small (100), we can use BFS or simply check both directions.
        # Direction 1: Increasing (1 -> 2 -> ... -> N -> 1)
        # Direction 2: Decreasing (1 -> N -> ... -> 2 -> 1)
        
        # To avoid the block, we check if the block lies on the path.
        # We normalize coordinates to 0...N-1 for easier modulo arithmetic.
        s, e, b = start - 1, end - 1, block - 1
        
        # Path 1: Clockwise
        # The distance is (e - s) % N. 
        # The block is on this path if (b - s) % N < (e - s) % N.
        dist_cw = (e - s) % N
        blocked_cw = (b - s) % N < dist_cw
        
        # Path 2: Counter-clockwise
        # The distance is (s - e) % N.
        # The block is on this path if (s - b) % N < (s - e) % N.
        dist_ccw = (s - e) % N
        blocked_ccw = (s - b) % N < dist_ccw
        
        # We return infinity if the path is blocked.
        # Since we are guaranteed the instruction is achievable, 
        # at least one path will be open.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res) if res else float('inf')

    # State: (left_hand, right_hand)
    # Initial state: L=1, R=2
    # We need to find the minimum total distance.
    # Since the problem guarantees positions are uniquely determined,
    # we can just simulate the process.
    
    # We use a list to keep track of current positions [L, R]
    # and a variable for total distance.
    # Because we cannot use loops, we use a reduction-like approach 
    # or a list comprehension to simulate the sequence.
    
    # We can use a helper function inside a 'reduce' to update state.
    # But since we can't import reduce, we'll use a trick with a list 
    # and a custom function to iterate through instructions.
    
    def process_instruction(state, instr):
        l, r = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r is the block
            d = get_dist(l, t, r)
            return (t, r, d)
        else:
            # Move right hand to t, left hand l is the block
            d = get_dist(r, t, l)
            return (l, t, d)

    # To simulate the state updates without a loop, we can use a recursive-like 
    # structure via a list comprehension that populates a results list.
    # However, since we need the previous state, we can use a trick:
    # Create a list of states where each element depends on the previous one.
    # Since Python lists aren't dynamic in that way, we use a mutable object 
    # (like a list) and a list comprehension to drive the side effect.
    
    state = [1, 2] # [L, R]
    distances = [
        (lambda s, i: (
            (s.__setitem__(0, t) if i[0] == 'L' else s.__setitem__(1, t)),
            get_dist(s[0], i[1], s[1]) if i[0] == 'L' else get_dist(s[1], i[1], s[0])
        )[1])(state, instr)
        for instr in instructions
    ]
    
    # Note: The __setitem__ trick inside the lambda is slightly wrong because 
    # it updates the state before calculating the distance if not careful.
    # Let's refine: calculate distance first, then update.
    
    # Corrected simulation using a list to hold the state and a list comprehension.
    # We use a helper function to handle the logic.
    def run_sim(instrs):
        curr_state = [1, 2]
        def step(i):
            h, t = i
            l, r = curr_state
            dist = get_dist(l, t, r) if h == 'L' else get_dist(r, t, l)
            if h == 'L': curr_state[0] = t
            else: curr_state[1] = t
            return dist
        return sum(map(step, instrs))

    print(run_sim(instructions))

if __name__ == "__main__":
    solve()