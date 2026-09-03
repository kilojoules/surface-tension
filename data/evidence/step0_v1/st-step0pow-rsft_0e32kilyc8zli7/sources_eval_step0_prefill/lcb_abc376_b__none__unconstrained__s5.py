import sys

def solve():
    # Read N and Q from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    l = 1
    r = 2
    total_ops = 0
    
    # Process instructions
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Move left hand from l to t, while right hand r is fixed.
            # The ring is a circle of N parts.
            # The right hand r acts as a barrier.
            # The left hand can move in two directions:
            # 1. Clockwise (increasing index)
            # 2. Counter-clockwise (decreasing index)
            # However, it cannot pass through part r.
            
            # Calculate distance clockwise and counter-clockwise
            # Distance is the number of steps.
            # To move from l to t without passing r:
            # We check if r is "between" l and t in either direction.
            
            # Normalize positions to 0-indexed for easier modulo arithmetic
            curr_l = l - 1
            curr_r = r - 1
            target_l = t - 1
            
            # Distance clockwise (l -> l+1 -> ...)
            # The path is l, (l+1)%N, ..., t.
            # It is blocked if r is any of the intermediate steps.
            # The number of steps is (target_l - curr_l) % N.
            dist_cw = (target_l - curr_l) % N
            # The path is blocked if curr_r is encountered.
            # curr_r is encountered if (curr_r - curr_l) % N < dist_cw.
            blocked_cw = (curr_r - curr_l) % N < dist_cw
            
            # Distance counter-clockwise (l -> l-1 -> ...)
            # The number of steps is (curr_l - target_l) % N.
            dist_ccw = (curr_l - target_l) % N
            # The path is blocked if (curr_l - curr_r) % N < dist_ccw.
            blocked_ccw = (curr_l - curr_r) % N < dist_ccw
            
            if not blocked_cw and not blocked_ccw:
                total_ops += min(dist_cw, dist_ccw)
            elif not blocked_cw:
                total_ops += dist_cw
            else:
                total_ops += dist_ccw
            
            l = t
            
        else: # h == 'R'
            # Move right hand from r to t, while left hand l is fixed.
            curr_l = l - 1
            curr_r = r - 1
            target_r = t - 1
            
            dist_cw = (target_r - curr_r) % N
            blocked_cw = (curr_l - curr_r) % N < dist_cw
            
            dist_ccw = (curr_r - target_r) % N
            blocked_ccw = (curr_r - curr_l) % N < dist_ccw
            
            if not blocked_cw and not blocked_ccw:
                total_ops += min(dist_cw, dist_ccw)
            elif not blocked_cw:
                total_ops += dist_cw
            else:
                total_ops += dist_ccw
                
            r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()