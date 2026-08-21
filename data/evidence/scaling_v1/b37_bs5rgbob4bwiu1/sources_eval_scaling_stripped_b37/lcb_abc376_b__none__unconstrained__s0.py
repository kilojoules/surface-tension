import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    instructions = []
    for i in range(Q):
        instructions.append((input_data[2 + 2*i], int(input_data[3 + 2*i])))

    def get_dist(start, end, obs, n):
        # Clockwise distance
        dist_cw = (end - start) % n
        # The clockwise path is blocked if the obstacle is "between" start and end.
        # Using 0-indexing for easier modulo math:
        s, e, o = start-1, end-1, obs-1
        # Clockwise path: s -> s+1 -> ... -> e
        # Obstacle o is on this path if (o - s) % n < (e - s) % n
        # Wait, (o-s)%n is the distance from s to o clockwise.
        # If this is less than the distance from s to e clockwise, it's blocked.
        blocked_cw = (o - s) % n < (e - s) % n if s != e else False
        # Actually, if s == e, dist is 0 and it's not blocked.
        # If s != e, the obstacle blocks the path if it's encountered before e.
        # Since we start at s, we check if o is reached before e.
        
        # Correct logic:
        # Path CW: s, (s+1)%n, ..., e. Blocked if o is one of these.
        # Since s is the start, the obstacle blocks if (o-s)%n < (e-s)%n.
        # Wait, if o == s, it's not blocking the *move* to the next cell.
        # But the problem says "other hand is not on the destination part".
        # So if the obstacle is at s, it doesn't block the first move.
        # If the obstacle is at o, it blocks the move to o.
        
        # Let's use a simpler check:
        # The only way a path is NOT blocked is if the obstacle is NOT on it.
        # The two paths together cover all nodes. 
        # One path is blocked if the obstacle is on it.
        # The obstacle is at 'o'. It is on the CW path if (o-s)%n < (e-s)%n.
        # EXCEPT if o == s, then it's not blocking the path to e.
        # But the problem says the other hand is at 'obs'. 
        # So the obstacle is at 'o'.
        
        # Let's use a helper:
        def check_blocked(s, e, o, direction):
            # direction 1: CW, -1: CCW
            # Distance to travel
            dist = (e - s) % n if direction == 1 else (s - e) % n
            # Distance to obstacle
            dist_o = (o - s) % n if direction == 1 else (s - o) % n
            return 0 < dist_o < dist

        d_cw = (end - start) % n
        d_ccw = (start - end) % n
        
        res = float('inf')
        if not check_blocked(start-1, end-1, obs-1, 1):
            res = min(res, d_cw)
        if not check_blocked(start-1, end-1, obs-1, -1):
            res = min(res, d_ccw)
        return res

    # Initial state
    current_states = {(1, 2): 0}

    for h, t in instructions:
        next_states = {}
        for (l, r), cost in current_states.items():
            if h == 'L':
                d = get_dist(l, t, r, N)
                if d != float('inf'):
                    next_states[(t, r)] = min(next_states.get((t, r), float('inf')), cost + d)
            else:
                d = get_dist(r, t, l, N)
                if d != float('inf'):
                    next_states[(l, t)] = min(next_states.get((l, t), float('inf')), cost + d)
        current_states = next_states

    print(min(current_states.values()))

solve()