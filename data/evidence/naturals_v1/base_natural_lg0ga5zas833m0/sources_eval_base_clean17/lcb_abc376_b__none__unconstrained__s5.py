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
        t = int(input_data[idx + 1])
        idx += 2
        
        if h == 'L':
            # Move left hand to t, right hand stays at r
            # The left hand can move clockwise or counter-clockwise
            # but it cannot pass through the position of the right hand.
            # The ring is 1 -> 2 -> ... -> N -> 1
            # The available path for L is the arc from l to t that does not contain r.
            
            # Distance clockwise from l to t
            dist_cw = (t - l + N) % N
            # Distance counter-clockwise from l to t
            dist_ccw = (l - t + N) % N
            
            # Check if r is on the clockwise path (l -> l+1 -> ... -> t)
            # r is on clockwise path if (r - l + N) % N < dist_cw
            # Special case: r is the destination t is handled by problem statement (T_i != r_i)
            # However, the constraint is that L cannot move TO r.
            # So we check if r is strictly between l and t on the clockwise path.
            
            # Let's calculate the distance from l to r clockwise
            dist_l_to_r_cw = (r - l + N) % N
            
            # If the clockwise distance to t is smaller than the distance to r,
            # or if r is not in the way of the clockwise path.
            # Actually, the simpler way:
            # Path 1: l -> l+1 -> ... -> t (clockwise)
            # Path 2: l -> l-1 -> ... -> t (counter-clockwise)
            # Path 1 is blocked if r is encountered.
            # r is encountered if (r-l+N)%N < (t-l+N)%N
            
            blocked_cw = (r - l + N) % N < (t - l + N) % N
            # Path 2 is blocked if r is encountered.
            # r is encountered if (l-r+N)%N < (l-t+N)%N
            blocked_ccw = (l - r + N) % N < (l - t + N) % N
            
            if not blocked_cw and not blocked_ccw:
                # Both paths are open? In a ring of N >= 3, if one hand is fixed,
                # only one path between two points is open unless the fixed hand
                # is not on the ring, which is impossible.
                # Actually, with one hand fixed, the ring becomes a line.
                # Only one path is available.
                pass 
            
            if not blocked_cw:
                total_ops += dist_cw
            else:
                total_ops += dist_ccw
            
            l = t
            
        else: # h == 'R'
            # Move right hand to t, left hand stays at l
            dist_cw = (t - r + N) % N
            dist_ccw = (r - t + N) % N
            
            # Path 1: r -> r+1 -> ... -> t (clockwise)
            # Blocked if l is encountered: (l-r+N)%N < (t-r+N)%N
            blocked_cw = (l - r + N) % N < (t - r + N) % N
            
            # Path 2: r -> r-1 -> ... -> t (counter-clockwise)
            # Blocked if l is encountered: (r-l+N)%N < (r-t+N)%N
            blocked_ccw = (r - l + N) % N < (r - t + N) % N
            
            if not blocked_cw:
                total_ops += dist_cw
            else:
                total_ops += dist_ccw
            
            r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()