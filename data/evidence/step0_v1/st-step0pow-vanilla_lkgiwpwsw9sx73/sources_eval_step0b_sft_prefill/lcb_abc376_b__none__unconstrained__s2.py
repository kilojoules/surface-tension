import sys
from collections import deque

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    current_l = 1
    current_r = 2
    total_ops = 0
    
    # Helper to calculate shortest distance between two parts on a ring
    # without passing through a forbidden part (the other hand)
    def get_dist(start, end, forbidden):
        # The ring is 1-indexed. We can treat it as a graph.
        # Since N is small (100), BFS is efficient to find the shortest path.
        # queue stores (current_node, distance)
        queue = deque([(start, 0)])
        visited = {start, forbidden}
        
        while queue:
            curr, dist = queue.popleft()
            if curr == end:
                return dist
            
            # Neighbors in a ring
            # Part i is connected to (i-1) and (i+1)
            # Handle wrap-around with 1-based indexing
            neighbors = [
                (curr - 2) % N + 1, 
                curr % N + 1
            ]
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    # Process instructions
    # The input is a list: [N, Q, H1, T1, H2, T2, ...]
    # Instructions start at index 2
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        
        if h == 'L':
            # Move left hand to t, right hand is forbidden
            dist = get_dist(current_l, t, current_r)
            total_ops += dist
            current_l = t
        else:
            # Move right hand to t, left hand is forbidden
            dist = get_dist(current_r, t, currentl := current_l)
            total_ops += dist
            current_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()