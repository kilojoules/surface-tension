import sys
from collections import deque

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    current_l = 1
    current_r = 2
    total_ops = 0
    
    # Helper to calculate shortest distance between two parts on a ring of size N
    # without passing through a forbidden part (the other hand)
    def get_dist(start, end, forbidden):
        # The ring is 1-indexed. We can treat it as 0-indexed for easier math.
        s, e, f = start - 1, end - 1, forbidden - 1
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One of these paths might be blocked by the forbidden part.
        
        # Path 1: s -> (s+1)%N -> ... -> e
        # Length is (e - s) % N
        # This path is blocked if any k in {s+1, ..., e-1} (mod N) is f.
        # More simply, f is on this path if (f - s) % N < (e - s) % N.
        
        dist_cw = (e - s) % N
        blocked_cw = (f - s) % N < dist_cw
        
        # Path 2: s -> (s-1)%N -> ... -> e
        # Length is (s - e) % N
        # This path is blocked if (s - f) % N < (s - e) % N.
        
        dist_ccw = (s - e) % N
        blocked_ccw = (s - f) % N < dist_ccw
        
        # We need the minimum distance of the non-blocked paths.
        # Since the problem guarantees the instruction is achievable, 
        # at least one path must be unblocked.
        
        results = []
        if not blocked_cw:
            results.append(dist_cw)
        if not blocked_ccw:
            results.append(dist_ccw)
            
        return min(results)

    # Process instructions
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            total_ops += get_dist(current_l, t, current_r)
            current_l = t
        else:
            total_ops += get_dist(current_r, t, current_l)
            current_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()