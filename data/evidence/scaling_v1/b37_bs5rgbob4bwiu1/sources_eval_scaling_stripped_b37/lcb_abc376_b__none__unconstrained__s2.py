import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    instructions = []
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        instructions.append((h, t))
        idx += 2

    # State: (left_hand, right_hand)
    # Initial state: (1, 2)
    # We want to find the minimum operations to satisfy Q instructions.
    # Since N and Q are small (100), and the other hand must remain stationary,
    # the movement is simply the shortest path on a ring avoiding the other hand.
    # On a ring of size N, if one hand is at 'other', the available space for the 
    # moving hand is a path of length N-1.
    # The distance between a and b on a path of length N-1 (where 'other' is the wall)
    # is simply the absolute difference if we "linearize" the ring starting from 'other'.
    
    def get_dist(start, end, other):
        # The movement is restricted to the arc that does not contain 'other'.
        # We can map the ring to a linear coordinate system where 'other' is at position 0.
        # The positions are (i - other) % N.
        # The available positions are 1, 2, ..., N-1.
        s = (start - other) % N
        if s == 0: s = N # Should not happen based on problem constraints
        e = (end - other) % N
        if e == 0: e = N # Should not happen
        
        # Since we cannot pass through 'other', we must move within the range [1, N-1].
        # The distance is simply |s - e|.
        return abs(s - e)

    # Current positions
    curr_l, curr_r = 1, 2
    total_ops = 0
    
    for h, t in instructions:
        if h == 'L':
            # Move left hand from curr_l to t, right hand stays at curr_r
            total_ops += get_dist(curr_l, t, curr_r)
            curr_l = t
        else:
            # Move right hand from curr_r to t, left hand stays at curr_l
            total_ops += get_dist(curr_r, t, curr_l)
            curr_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()