import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    curr_l = 1
    curr_r = 2
    
    # Helper to calculate shortest distance between two parts on a ring
    # without passing through a forbidden part (the other hand)
    def get_dist(start, end, forbidden):
        # The ring is 1-indexed. We can treat it as 0-indexed for easier math.
        s = start - 1
        e = end - 1
        f = forbidden - 1
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One of these paths might be blocked by the forbidden part.
        
        # Path 1: s -> (s+1)%N -> (s+2)%N ... -> e
        # Length is (e - s) mod N
        dist1 = (e - s) % N
        # This path is blocked if f is between s and e (exclusive)
        # f is on this path if (f - s) % N < dist1
        blocked1 = (f - s) % N < dist1
        
        # Path 2: s -> (s-1)%N -> (s-2)%N ... -> e
        # Length is (s - e) mod N
        dist2 = (s - e) % N
        # This path is blocked if f is between s and e (exclusive)
        # f is on this path if (s - f) % N < dist2
        blocked2 = (s - f) % N < dist2
        
        # We want the minimum distance of the unblocked paths.
        # Since it's guaranteed that the instruction is achievable, 
        # at least one path must be unblocked.
        
        res = float('inf')
        if not blocked1:
            res = min(res, dist1)
        if not blocked2:
            res = min(res, dist2)
            
        return res

    # Process instructions
    total_ops = 0
    ptr = 2
    for _ in range(Q):
        h = input_data[ptr]
        t = int(input_data[ptr+1])
        ptr += 2
        
        if h == 'L':
            # Move left hand to t, right hand is forbidden
            dist = get_dist(curr_l, t, curr_r)
            total_ops += dist
            curr_l = t
        else:
            # Move right hand to t, left hand is forbidden
            dist = get_dist(curr_r, t, curr_l)
            total_ops += dist
            curr_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()