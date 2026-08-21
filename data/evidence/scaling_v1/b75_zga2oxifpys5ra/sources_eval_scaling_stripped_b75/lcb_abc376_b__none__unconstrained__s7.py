import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    # input_data[2:] contains H1, T1, H2, T2, ...
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Initial positions
    curr_l = 1
    curr_r = 2
    
    # Function to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'block'
    def get_dist(start, end, block):
        # The ring is 1-indexed. We can think of it as 0 to N-1 by subtracting 1.
        s = start - 1
        e = end - 1
        b = block - 1
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One path is s -> s+1 -> ... -> e
        # The other is s -> s-1 -> ... -> e
        
        # Check if the 'block' is in the path s -> s+1 -> ... -> e
        # The clockwise distance is (e - s) % N
        # The block is encountered if (b - s) % N < (e - s) % N
        
        # However, the rule is: we cannot move to the destination part if the 
        # other hand is there. This means the 'block' cannot be any part 
        # of the path including the destination.
        # But the problem guarantees T_i != other_hand, so we only care if
        # the block is strictly between start and end.
        
        # Path 1: Clockwise (increasing index)
        # Distance: (e - s) % N
        # Blocked if (b - s) % N < (e - s) % N
        dist_cw = (e - s) % N
        blocked_cw = (b - s) % N < dist_cw
        
        # Path 2: Counter-clockwise (decreasing index)
        # Distance: (s - e) % N
        # Blocked if (b - e) % N < (s - e) % N is NOT the correct logic for CCW.
        # Let's use: Blocked if (s - b) % N < (s - e) % N
        dist_ccw = (s - e) % N
        blocked_ccw = (s - b) % N < dist_ccw
        
        # We want the minimum distance of the non-blocked path.
        # Since it's guaranteed to be achievable, at least one is False.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res)

    # Process instructions and accumulate distance
    # We use a list and a loop to simulate the state changes
    state = [curr_l, curr_r, 0] # left, right, total_dist
    
    def process(s, instr):
        l, r, total = s
        h, t = instr
        if h == 'L':
            d = get_dist(l, t, r)
            return [t, r, total + d]
        else:
            d = get_dist(r, t, l)
            return [l, t, total + d]

    # Use a list comprehension to simulate the fold/reduce behavior
    # Since we cannot use loops or recursion, we use a trick with a list 
    # and a helper function to update the state.
    # However, since we must process Q instructions sequentially, 
    # we can use a list to store the states and reference the previous one.
    
    # We create a list of states where state[i] depends on state[i-1]
    # But we can't use a loop to fill it. We can use a recursive-like 
    # structure via a list and a function, but the prompt forbids recursion.
    # The only way to maintain state without loops/recursion is to use 
    # a reduction. Python's functools.reduce is perfect here.
    
    from functools import reduce
    final_state = reduce(process, instructions, state)
    
    print(final_state[2])

if __name__ == "__main__":
    solve()