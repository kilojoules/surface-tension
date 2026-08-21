import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    l = 1
    r = 2
    total_ops = 0
    
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Left hand moves from l to t, Right hand stays at r
            # The ring is a cycle of N nodes.
            # To move from l to t without passing through r:
            # There are two paths on a circle: clockwise and counter-clockwise.
            # One path might be blocked by r.
            
            # Let's represent positions as 0 to N-1 for easier modulo arithmetic
            curr_l = l - 1
            curr_r = r - 1
            target_l = t - 1
            
            # Path 1: Increasing index (curr_l -> curr_l + 1 -> ...)
            # Path 2: Decreasing index (curr_l -> curr_l - 1 -> ...)
            
            # Distance clockwise (increasing)
            dist_cw = (target_l - curr_l + N) % N
            # Distance counter-clockwise (decreasing)
            dist_ccw = (curr_l - target_l + N) % N
            
            # Check if r is in the clockwise path
            # r is in the path if (r - curr_l + N) % N <= dist_cw AND r != curr_l
            # Actually, the rule is: we can't move to r.
            # If we move clockwise, we visit (curr_l + i) % N for i = 1 to dist_cw.
            # If any of these equals curr_r, this path is blocked.
            blocked_cw = False
            for i in range(1, dist_cw + 1):
                if (curr_l + i) % N == curr_r:
                    blocked_cw = True
                    break
            
            # Similarly for counter-clockwise
            blocked_ccw = False
            for i in range(1, dist_ccw + 1):
                if (curr_l - i + N) % N == curr_r:
                    blocked_ccw = True
                    break
            
            if not blocked_cw and not blocked_ccw:
                total_ops += min(dist_cw, dist_ccw)
            elif not blocked_cw:
                total_ops += dist_cw
            elif not blocked_ccw:
                total_ops += dist_ccw
            # It's guaranteed that the instruction is achievable.
            
            l = t
            
        else: # h == 'R'
            # Right hand moves from r to t, Left hand stays at l
            curr_l = l - 1
            curr_r = r - 1
            target_r = t - 1
            
            dist_cw = (target_r - curr_r + N) % N
            dist_ccw = (curr_r - target_r + N) % N
            
            blocked_cw = False
            for i in range(1, dist_cw + 1):
                if (curr_r + i) % N == curr_l:
                    blocked_cw = True
                    break
            
            blocked_ccw = False
            for i in range(1, dist_ccw + 1):
                if (curr_r - i + N) % N == curr_l:
                    blocked_ccw = True
                    break
            
            if not blocked_cw and not blocked_ccw:
                total_ops += min(dist_cw, dist_ccw)
            elif not blocked_cw:
                total_ops += dist_cw
            elif not blocked_ccw:
                total_ops += dist_ccw
            
            r = t

    print(total_ops)

if __name__ == "__main__":
    solve()