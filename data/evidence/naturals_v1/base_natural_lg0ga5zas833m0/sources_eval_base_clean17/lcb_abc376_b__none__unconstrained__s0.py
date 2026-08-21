import sys

def solve():
    # Read N and Q from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, q = map(int, line1)
    except ValueError:
        return

    # Initial positions: Left hand at 1, Right hand at 2
    l_pos = 1
    r_pos = 2
    total_ops = 0

    for _ in range(q):
        line = sys.stdin.readline().split()
        if not line:
            break
        h = line[0]
        t = int(line[1])

        if h == 'L':
            # Left hand moves from l_pos to t, while Right hand stays at r_pos
            # There are two paths on the ring: clockwise and counter-clockwise
            # The restriction is that the left hand cannot pass through r_pos
            
            # Calculate distance clockwise (1 -> 2 -> ... -> N -> 1)
            # To avoid r_pos, we check if r_pos is on the path.
            # Path 1: l_pos -> (l_pos+1)%N ... -> t
            # Path 2: l_pos -> (l_pos-1)%N ... -> t
            
            # Simplest way to find the distance without crossing r_pos:
            # The ring is a cycle. The other hand (r_pos) splits the cycle into a linear path.
            # The distance is the distance from l_pos to t on the graph where the edge 
            # containing r_pos is removed? No, r_pos is a node.
            # The path cannot enter node r_pos.
            # So we are moving on a graph with nodes {1, ..., N} \ {r_pos}.
            # This is a line of N-1 nodes.
            
            # Let's map the ring to a linear coordinate system relative to r_pos.
            # Let r_pos be the "barrier".
            # The available parts are those in the range (r_pos, r_pos) excluding r_pos.
            # We can represent positions as (pos - r_pos) % N.
            # r_pos becomes 0. Other positions are 1, 2, ..., N-1.
            # The movement is now on a line from 1 to N-1.
            
            curr_rel = (l_pos - r_pos) % n
            if curr_rel == 0: curr_rel = n # Should not happen based on problem statement
            target_rel = (t - r_pos) % n
            if target_rel == 0: target_rel = n # Should not happen
            
            # In the relative system, the distance is simply the absolute difference
            # because we cannot cross the 0 (r_pos) point.
            # Note: (x - r_pos) % n maps r_pos to 0, and others to 1...N-1.
            # The distance is |target_rel - curr_rel|.
            
            # Correcting relative mapping:
            # Let's use: f(x) = (x - r_pos - 1) % n
            # This maps r_pos to n-1, and the path from r_pos+1 to r_pos-1 to 0...n-2.
            # Actually, simpler:
            # The distance is the number of steps from l_pos to t without passing through r_pos.
            # The number of steps is |l_pos - t| if we don't cross the N-1 boundary.
            # But the ring is circular.
            
            # Let's use the property: the distance is the absolute difference of 
            # their positions in a linear array where r_pos is at the boundary.
            def get_dist(start, end, barrier, N):
                # Shift everything so barrier is at "position 0" (or N)
                # New positions: (x - barrier) % N
                # This maps barrier to 0, and others to 1, 2, ..., N-1.
                # Since we can't pass through barrier, we stay within 1...N-1.
                s = (start - barrier) % N
                if s == 0: s = N # Should not happen
                e = (end - barrier) % N
                if e == 0: e = N # Should not happen
                return abs(s - e)

            dist = get_dist(l_pos, t, r_pos, n)
            total_ops += dist
            l_pos = t
            
        else: # h == 'R'
            # Right hand moves from r_pos to t, while Left hand stays at l_pos
            def get_dist(start, end, barrier, N):
                s = (start - barrier) % N
                if s == 0: s = N
                e = (end - barrier) % N
                if e == 0: e = N
                return abs(s - e)
            
            dist = get_dist(r_pos, t, l_pos, n)
            total_ops += dist
            r_pos = t

    print(total_ops)

if __name__ == "__main__":
    solve()