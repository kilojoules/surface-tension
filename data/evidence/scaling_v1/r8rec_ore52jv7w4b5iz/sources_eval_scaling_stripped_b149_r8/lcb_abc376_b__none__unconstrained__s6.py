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
    # without passing through a forbidden point 'obstacle'
    def get_dist(start, end, obstacle):
        if start == end:
            return 0
        
        # There are two directions on the ring: clockwise and counter-clockwise.
        # We need to check if the obstacle lies on the path.
        
        # Path 1: Increasing indices (with wrap around)
        # The points visited are (start + k) % N
        # The obstacle is hit if (obstacle - start) % N < (end - start) % N
        # Wait, a simpler way:
        # Distance clockwise from start to end is (end - start) % N
        # The obstacle is in the way if (obstacle - start) % N < (end - start) % N
        
        dist_cw = (end - start) % N
        # We use 0-indexing for modulo, so we adjust 1-N to 0-(N-1)
        # But since we use (a-b)%N, the offset cancels out.
        
        # Check if obstacle is between start and end clockwise
        # Normalize everything to 0...N-1 for the check
        s, e, o = (start-1)%N, (end-1)%N, (obstacle-1)%N
        
        # Clockwise distance
        d_cw = (e - s) % N
        # Obstacle is encountered if (o-s)%N < d_cw
        blocked_cw = (o - s) % N < d_cw
        
        # Counter-clockwise distance
        d_ccw = (s - e) % N
        # Obstacle is encountered if (o-e)%N < d_ccw
        blocked_ccw = (o - e) % N < d_ccw
        
        # We want the minimum of the non-blocked paths.
        # The problem guarantees the instruction is achievable.
        res = []
        if not blocked_cw: res.append(d_cw)
        if not blocked_ccw: res.append(d_ccw)
        
        return min(res)

    # State: (left_hand, right_hand)
    # Initial state: (1, 2)
    # We need to find the sequence of states that minimizes total distance.
    # Since the problem says positions are uniquely determined, 
    # we can just simulate greedily/iteratively.
    
    current_l, current_r = 1, 2
    total_dist = 0
    
    for h, t in instructions:
        if h == 'L':
            # Move left hand from current_l to t, right hand at current_r is obstacle
            total_dist += get_dist(current_l, t, current_r)
            current_l = t
        else:
            # Move right hand from current_r to t, left hand at current_l is obstacle
            total_dist += get_dist(current_r, t, current_l)
            current_r = t
            
    print(total_dist)

if __name__ == "__main__":
    solve()