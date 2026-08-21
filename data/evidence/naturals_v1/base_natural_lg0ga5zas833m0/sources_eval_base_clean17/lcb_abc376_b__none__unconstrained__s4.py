import sys

def solve():
    # Read N and Q from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    l_pos = 1
    r_pos = 2
    total_ops = 0
    
    ptr = 2
    for _ in range(Q):
        h = input_data[ptr]
        t = int(input_data[ptr + 1])
        ptr += 2
        
        if h == 'L':
            # Move left hand to t while right hand stays at r_pos
            # Left hand can move in two directions on the ring: clockwise or counter-clockwise
            # However, it cannot pass through r_pos.
            # The ring is [1, 2, ..., N].
            # The path from l_pos to t must not contain r_pos.
            
            # Distance clockwise (increasing index)
            # If r_pos is "between" l_pos and t in the clockwise direction, 
            # the clockwise path is blocked.
            # Let's represent positions as 0 to N-1 for easier modulo arithmetic.
            l = l_pos - 1
            r = r_pos - 1
            target = t - 1
            
            # Check if target is reachable clockwise without hitting r
            # Clockwise: l -> l+1 -> ... -> target
            # The path is blocked if r is in the range (l, target) mod N.
            
            # Simplest way to find shortest path avoiding r:
            # There are only two paths on a ring. One is blocked by r, the other is not.
            # Since it's guaranteed to be achievable, one path is always open.
            
            # Distance clockwise: (target - l + N) % N
            # Distance counter-clockwise: (l - target + N) % N
            
            # Is the clockwise path blocked?
            # The clockwise path uses nodes (l+1)%N, (l+2)%N ... target.
            # It's blocked if any of these equals r.
            is_blocked_cw = False
            curr = l
            while curr != target:
                curr = (curr + 1) % N
                if curr == r:
                    is_blocked_cw = True
                    break
            
            if not is_blocked_cw:
                dist = (target - l + N) % N
            else:
                dist = (l - target + N) % N
            
            total_ops += dist
            l_pos = t
            
        else: # h == 'R'
            # Move right hand to t while left hand stays at l_pos
            l = l_pos - 1
            r = r_pos - 1
            target = t - 1
            
            is_blocked_cw = False
            curr = r
            while curr != target:
                curr = (curr + 1) % N
                if curr == l:
                    is_blocked_cw = True
                    break
            
            if not is_blocked_cw:
                dist = (target - r + N) % N
            else:
                dist = (r - target + N) % N
                
            total_ops += dist
            r_pos = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()