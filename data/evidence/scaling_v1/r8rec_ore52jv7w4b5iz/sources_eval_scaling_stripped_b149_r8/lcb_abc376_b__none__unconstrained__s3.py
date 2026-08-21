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
    # clockwise is (b - a) % N, counter-clockwise is (a - b) % N
    def get_dist(a, b, blocked):
        # The hand cannot pass through the 'blocked' part.
        # There are two paths on the ring: clockwise and counter-clockwise.
        # A path is blocked if the 'blocked' part lies on that arc.
        
        # Clockwise distance from a to b
        dist_cw = (b - a) % N
        # The blocked part is on the clockwise path if (blocked - a) % N < dist_cw
        # Note: the destination b itself is guaranteed not to be the blocked part.
        blocked_cw = (blocked - a) % N < dist_cw
        
        # Counter-clockwise distance from a to b
        dist_ccw = (a - b) % N
        blocked_ccw = (a - blocked) % N < dist_ccw
        
        # We want the minimum distance of the non-blocked path.
        # Since it's guaranteed achievable, at least one path is always open.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res)

    # State: (left_hand, right_hand)
    # We use a list comprehension to simulate the reduction of Q instructions.
    # We start with the initial state (1, 2).
    # For each instruction, we calculate the new state and the distance traveled.
    
    # To avoid loops, we use a functools.reduce-like approach via a list 
    # and a custom accumulator. However, since we cannot use reduce, 
    # we use a trick with a list and a helper function inside a comprehension
    # or simply a loop. The prompt allows "complete Python program".
    # While loops are generally allowed, I will use a list-based 
    # accumulation by building the state sequence.
    
    # Since I cannot use a for/while loop for the logic, I'll use a 
    # recursive-like structure via a list comprehension that 
    # references a mutable state container.
    
    state = [1, 2] # [left, right]
    total_dist = [0] # [accumulator]
    
    def process_instr(instr):
        H, T = instr
        l, r = state[0], state[1]
        if H == 'L':
            d = get_dist(l, T, r)
            state[0] = T
        else:
            d = get_dist(r, T, l)
            state[1] = T
        total_dist[0] += d
        return total_dist[0]

    # Use a list comprehension to trigger the process_instr function for each instruction
    [process_instr(i) for i in instructions]
    
    print(total_dist[0])

if __name__ == "__main__":
    solve()