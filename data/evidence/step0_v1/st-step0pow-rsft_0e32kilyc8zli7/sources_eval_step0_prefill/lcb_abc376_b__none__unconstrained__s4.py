import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Current positions of hands
    l = 1
    r = 2
    total_ops = 0
    
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Move left hand from l to t, right hand r is fixed.
            # The ring is 1...N. The right hand r blocks one position.
            # The left hand can move in two directions:
            # 1. Clockwise (increasing index)
            # 2. Counter-clockwise (decreasing index)
            # However, it cannot pass through r.
            
            # Since it's a ring, we can think of positions as 0 to N-1
            # but the problem uses 1 to N.
            # Let's normalize to 0...N-1 for easier modulo arithmetic.
            curr_l = l - 1
            curr_r = r - 1
            target_l = t - 1
            
            # The right hand r divides the ring into a path.
            # The only way to get from curr_l to target_l without crossing curr_r
            # is to move along the arc that does not contain curr_r.
            
            # Calculate distance clockwise (increasing)
            # Distance from curr_l to target_l clockwise:
            dist_cw = (target_l - curr_l + N) % N
            # Does the clockwise path from curr_l to target_l contain curr_r?
            # The path is curr_l, (curr_l + 1)%N, ..., target_l.
            # curr_r is on this path if (curr_r - curr_l + N) % N <= dist_cw.
            
            if (curr_r - curr_l + N) % N <= dist_cw:
                # Clockwise is blocked, must go counter-clockwise.
                # Distance counter-clockwise is N - dist_cw.
                total_ops += (N - dist_cw)
            else:
                # Clockwise is clear.
                total_ops += dist_cw
            
            l = t
        else:
            # Move right hand from r to t, left hand l is fixed.
            curr_l = l - 1
            curr_r = r - 1
            target_r = t - 1
            
            dist_cw = (target_r - curr_r + N) % N
            # Check if curr_l is on the clockwise path from curr_r to target_r.
            if (curr_l - curr_r + N) % N <= dist_cw:
                # Clockwise is blocked, go counter-clockwise.
                total_ops += (N - dist_cw)
            else:
                # Clockwise is clear.
                total_ops += dist_cw
                
            r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()