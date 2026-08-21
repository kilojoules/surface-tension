import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Current positions of hands
    # Left hand (L) starts at 1, Right hand (R) starts at 2
    cur_l = 1
    cur_r = 2
    total_ops = 0
    
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Move left hand to T, right hand stays at cur_r
            # Left hand can move in two directions on the ring:
            # Clockwise: (cur_l + 1), (cur_l + 2)...
            # Counter-clockwise: (cur_l - 1), (cur_l - 2)...
            # However, it cannot step on cur_r.
            
            # Since the ring is small (N <= 100), we can use BFS to find the shortest path
            # from cur_l to t without passing through cur_r.
            dist = get_dist(cur_l, t, cur_r, N)
            total_ops += dist
            cur_l = t
        else:
            # Move right hand to T, left hand stays at cur_l
            dist = get_dist(cur_r, t, cur_l, N)
            total_ops += dist
            cur_r = t
            
    print(total_ops)

def get_dist(start, end, obstacle, N):
    if start == end:
        return 0
    
    # BFS to find the shortest path on the ring
    # Nodes are 1 to N
    queue = [(start, 0)]
    visited = {start}
    
    head = 0
    while head < len(queue):
        curr, d = queue[head]
        head += 1
        
        # Possible moves: clockwise and counter-clockwise
        # Clockwise: (curr % N) + 1
        # Counter-clockwise: (curr - 2) % N + 1
        moves = [
            (curr % N) + 1,
            (curr - 2) % N + 1
        ]
        
        for move in moves:
            if move == end:
                return d + 1
            if move != obstacle and move not in visited:
                visited.add(move)
                queue.append((move, d + 1))
                
    return -1 # Should not happen based on problem guarantee

if __name__ == "__main__":
    solve()