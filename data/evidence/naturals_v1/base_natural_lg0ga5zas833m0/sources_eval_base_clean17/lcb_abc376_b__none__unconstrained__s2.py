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

    # Initial positions of hands
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
            # We need to move the left hand to target t, while right hand stays at r_pos.
            # The ring is 1 to N.
            # The right hand acts as a barrier at r_pos.
            # There are two paths from l_pos to t on a ring: clockwise and counter-clockwise.
            # However, one path might be blocked by r_pos.
            
            # To find the shortest path from l_pos to t that doesn't pass through r_pos:
            # We can model the ring as a graph or simply check the two directions.
            
            # Distance clockwise (increasing)
            # If we move from l to t by increasing index (with wrap around), 
            # does it pass through r_pos?
            
            # Let's simplify: the available space is a linear segment from 
            # (r_pos + 1) mod N to (r_pos - 1) mod N.
            # Since N is small (100), we can use BFS to find the shortest path.
            
            dist = get_shortest_path(l_pos, t, r_pos, n)
            total_ops += dist
            l_pos = t
        else:
            # Move right hand to target t, while left hand stays at l_pos.
            dist = get_shortest_path(r_pos, t, l_pos, n)
            total_ops += dist
            r_pos = t

    print(total_ops)

def get_shortest_path(start, end, barrier, n):
    if start == end:
        return 0
    
    # BFS to find the shortest path on the ring avoiding the barrier
    queue = [(start, 0)]
    visited = {start}
    
    idx = 0
    while idx < len(queue):
        curr, d = queue[idx]
        idx += 1
        
        # Possible moves: (curr + 1) and (curr - 1) with wrap around
        for move in [1, -1]:
            nxt = curr + move
            if nxt > n: nxt = 1
            if nxt < 1: nxt = n
            
            if nxt != barrier and nxt not in visited:
                if nxt == end:
                    return d + 1
                visited.add(nxt)
                queue.append((nxt, d + 1))
    return -1 # Should not happen based on problem guarantee

if __name__ == "__main__":
    solve()