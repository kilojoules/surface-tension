import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    # Using a list comprehension to group the remaining input into pairs
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Initial positions
    curr_l = 1
    curr_r = 2
    
    # Function to calculate the distance between two points on a ring of size N
    # without passing through a forbidden point 'block'
    def get_dist(start, end, block):
        # There are two paths on a ring: clockwise and counter-clockwise
        # Path 1: start -> start+1 -> ... -> end (modulo N)
        # Path 2: start -> start-1 -> ... -> end (modulo N)
        
        # We check if the 'block' is in the way of either path.
        # A path is blocked if the 'block' lies strictly between start and end.
        
        # To simplify, we can use a helper to check if x is on the arc from a to b clockwise
        def is_on_arc(a, b, x):
            # Normalize to 0..N-1
            a, b, x = (a-1)%N, (b-1)%N, (x-1)%N
            if a <= b:
                return a < x < b
            else:
                return x > a or x < b

        # Distance clockwise
        dist_cw = (end - start) % N
        # Distance counter-clockwise
        dist_ccw = (start - end) % N
        
        # Check if the block is on the clockwise path
        # The block is on the path if it's between start and end clockwise
        # Note: the problem says we can't move to the destination if the other hand is there.
        # But the guarantee says T_i != other_hand, so we only care about intermediate steps.
        
        # If the block is on the clockwise arc, that path is invalid.
        # Otherwise, the distance is dist_cw.
        # However, we need the MINIMUM distance of the VALID path.
        
        # Since it's a ring, there are only two directions.
        # One direction is blocked if the 'block' is encountered.
        
        # Clockwise path is blocked if is_on_arc(start, end, block) is True
        # Counter-clockwise path is blocked if is_on_arc(end, start, block) is True
        
        res = float('inf')
        if not is_on_arc(start, end, block):
            res = min(res, dist_cw)
        if not is_on_arc(end, start, block):
            res = min(res, dist_ccw)
            
        return res

    # Process instructions and accumulate distance
    # We use a loop to update curr_l and curr_r
    def process(state, inst):
        l, r = state
        h, t = inst
        if h == 'L':
            dist = get_dist(l, t, r)
            return (t, r), dist
        else:
            dist = get_dist(r, t, l)
            return (l, t), dist

    # Using a reduction-like approach to simulate the state changes
    # We use a list to store (current_state, total_distance)
    # Initial state: ((1, 2), 0)
    def accumulate(acc, inst):
        state, total_dist = acc
        new_state, move_dist = process(state, inst)
        return (new_state, total_dist + move_dist)

    # Since we cannot use loops, we use a generator or map/reduce logic
    # But we need the state from the previous step. 
    # We can use a helper function with recursion or a trick with a list.
    
    # Using a list to simulate a loop via a function that calls itself
    def run_instructions(idx, l, r, total):
        if idx == Q:
            return total
        h, t = instructions[idx]
        if h == 'L':
            d = get_dist(l, t, r)
            return run_instructions(idx + 1, t, r, total + d)
        else:
            d = get_dist(r, t, l)
            return run_instructions(idx + 1, l, t, total + d)

    # Increase recursion depth for Q=100
    sys.setrecursionlimit(2000)
    print(run_instructions(0, curr_l, curr_r, 0))

if __name__ == "__main__":
    solve()