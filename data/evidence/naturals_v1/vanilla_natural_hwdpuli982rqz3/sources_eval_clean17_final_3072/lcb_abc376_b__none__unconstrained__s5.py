import sys
from collections import deque

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    current_l = 1
    current_r = 2
    
    # Helper to calculate shortest distance between two parts on a ring
    # without passing through a forbidden part
    def get_dist(start, end, forbidden):
        if start == end:
            return 0
        
        # BFS to find shortest path on the ring avoiding 'forbidden'
        # Since N is small (100), BFS is efficient.
        # queue stores (current_node, distance)
        queue = deque([(start, 0)])
        visited = {start, forbidden}
        
        while queue:
            u, d = queue.popleft()
            if u == end:
                return d
            
            # Neighbors on the ring
            # Part i is connected to (i-1) and (i+1)
            # Handle wrap-around with 1-based indexing
            neighbors = [
                (u - 2) % N + 1, 
                u % N + 1
            ]
            
            for v in neighbors:
                if v not in visited:
                    visited.add(v)
                    queue.append((v, d + 1))
        return float('inf')

    # Process instructions
    instructions = input_data[2:]
    total_ops = 0
    
    # We use a list to keep track of current positions because we can't use 
    # nonlocal/global easily inside a loop without recursion or helper functions
    pos = [current_l, current_r]
    
    for i in range(0, len(instructions), 2):
        h = instructions[i]
        t = int(instructions[i+1])
        
        if h == 'L':
            # Move left hand to t, right hand is forbidden
            dist = get_dist(pos[0], t, pos[1])
            total_ops += dist
            pos[0] = t
        else:
            # Move right hand to t, left hand is forbidden
            dist = get_dist(pos[1], t, pos[0])
            total_ops += dist
            pos[1] = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()