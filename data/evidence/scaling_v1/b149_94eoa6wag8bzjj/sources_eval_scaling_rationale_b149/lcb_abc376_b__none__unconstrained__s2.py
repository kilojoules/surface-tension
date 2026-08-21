import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Helper to calculate distance between two points on a ring of size N
    # given that a certain point 'block' cannot be passed.
    # The distance is the number of edges.
    def get_dist(start, end, block, n):
        # There are two directions on the ring: clockwise and counter-clockwise.
        # One direction is blocked by the 'block' hand.
        # We need to find the distance of the path that does NOT contain 'block'.
        
        # Normalize coordinates to 0...N-1
        s, e, b = start - 1, end - 1, block - 1
        
        # The two possible paths are:
        # 1. s -> (s+1)%N -> ... -> e
        # 2. s -> (s-1)%N -> ... -> e
        
        # Path 1 distance
        dist1 = (e - s) % n
        # Path 1 contains 'block' if (block - s) % n < dist1
        # Special case: if s == b, it's technically starting at the block, 
        # but the problem guarantees T != other_hand.
        blocked1 = (b - s) % n < dist1 if s != b else False
        
        # Path 2 distance
        dist2 = (s - e) % n
        blocked2 = (s - b) % n < dist2 if s != b else False
        
        # Since it's guaranteed achievable, one path must be open.
        # We want the minimum distance of the non-blocked path.
        # Actually, only one path can be non-blocked unless N=2 (but N>=3).
        return dist1 if not blocked1 else dist2

    # State: (current_l, current_r, total_dist)
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

    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()