import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Current positions of hands
    # Left hand is at 1, Right hand is at 2
    l_pos = 1
    r_pos = 2
    total_ops = 0
    
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Move left hand from l_pos to t, while right hand stays at r_pos
            # The ring is 1...N. Distance is calculated on a circle.
            # However, the other hand (r_pos) acts as a barrier.
            # We can move clockwise or counter-clockwise, but cannot cross r_pos.
            
            # Distance clockwise: (t - l_pos + N) % N
            # Distance counter-clockwise: (l_pos - t + N) % N
            
            # To check if a path is blocked by r_pos:
            # Path clockwise from l to t: l, l+1, ..., t (mod N)
            # Path counter-clockwise from l to t: l, l-1, ..., t (mod N)
            
            # Since we are on a ring of N parts, and one part is blocked,
            # there is only one path that doesn't pass through the blocked part
            # unless the blocked part is not on the path at all.
            
            # Let's calculate distance and check if the blocked position is "between" them.
            # More simply: a path is blocked if the blocked position is on the arc.
            
            # Clockwise distance
            dist_cw = (t - l_pos + N) % N
            # Is r_pos on the clockwise arc from l_pos to t?
            # r_pos is on the clockwise arc if (r_pos - l_pos + N) % N < dist_cw
            # but we must be careful with the boundary.
            # Actually, the only way to get from l to t is to go the way that doesn't hit r.
            
            # Let's use a simpler approach:
            # The ring is divided into two arcs by l_pos and r_pos.
            # To move l to t without crossing r, l must move within the arc that contains t.
            # If t is the same as l, distance is 0.
            if l_pos == t:
                dist = 0
            else:
                # There are two paths: clockwise and counter-clockwise.
                # One path is (l_pos -> l_pos+1 -> ... -> t)
                # Other path is (l_pos -> l_pos-1 -> ... -> t)
                # Check if r_pos is on the clockwise path:
                # The clockwise path consists of nodes (l_pos + k - 1) % N + 1 for k=0 to dist_cw
                # r_pos is on this path if (r_pos - l_pos + N) % N < dist_cw
                
                is_blocked_cw = False
                # Check if r_pos is strictly between l_pos and t clockwise
                # (r_pos - l_pos + N) % N is the clockwise distance from l to r.
                if 0 < (r_pos - l_pos + N) % N < dist_cw:
                    is_blocked_cw = True
                
                if is_blocked_cw:
                    # Must go counter-clockwise
                    dist = (l_pos - t + N) % N
                else:
                    # Can go clockwise (this is the shorter path if it's not blocked)
                    # Wait, the problem asks for the minimum operations.
                    # If both paths are available, take the minimum. 
                    # But only one path is available because r_pos is a barrier.
                    # Actually, if r_pos is not on the clockwise path, the clockwise path is clear.
                    # If r_pos is not on the counter-clockwise path, that path is clear.
                    # Since r_pos != l_pos and r_pos != t, r_pos must be on exactly one of the two arcs.
                    
                    # The arc that does NOT contain r_pos is the only viable path.
                    dist = dist_cw
                    # Check if the counter-clockwise path is also clear? 
                    # No, because r_pos is on the ring. One of the two arcs must contain r_pos.
                    # Unless N=2, but N >= 3.
                    # Let's re-evaluate:
                    # Clockwise distance: cw = (t - l_pos + N) % N
                    # Counter-clockwise distance: ccw = (l_pos - t + N) % N
                    # If r_pos is on the clockwise arc, we must go ccw.
                    # If r_pos is on the ccw arc, we must go cw.
                    # r_pos is on clockwise arc if (r_pos - l_pos + N) % N < cw
                    # r_pos is on ccw arc if (l_pos - r_pos + N) % N < ccw
                    
                    # Let's use this:
                    cw = (t - l_pos + N) % N
                    ccw = (l_pos - t + N) % N
                    if (r_pos - l_pos + N) % N < cw:
                        dist = ccw
                    else:
                        dist = cw
            
            total_ops += dist
            l_pos = t
            
        else: # h == 'R'
            if r_pos == t:
                dist = 0
            else:
                cw = (t - r_pos + N) % N
                ccw = (r_pos - t + N) % N
                if (l_pos - r_pos + N) % N < cw:
                    dist = ccw
                else:
                    dist = cw
            total_ops += dist
            r_pos = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()