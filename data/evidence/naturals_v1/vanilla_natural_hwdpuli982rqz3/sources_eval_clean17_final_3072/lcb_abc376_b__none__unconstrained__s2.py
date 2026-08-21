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
    
    # BFS function to find shortest path for one hand while the other is stationary
    def get_dist(start, target, stationary):
        # The ring is 1-indexed. We use a queue for BFS.
        # distance dictionary stores {node: dist}
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            curr, dist = queue.popleft()
            if curr == target:
                return dist
            
            # Neighbors in a ring of size N
            # Part i is connected to (i-1) and (i+1)
            # Handle wrap-around using 1-based indexing logic
            neighbors = [
                (curr - 2) % N + 1, 
                curr % N + 1
            ]
            
            for neighbor in neighbors:
                if neighbor != stationary and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    # Process instructions
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx + 1])
        idx += 2
        
        if h == 'L':
            # Move left hand to t, right hand stays at current_r
            dist = get_dist(current_l, t, current_r)
            total_ops += dist
            current_l = t
        else:
            # Move right hand to t, left hand stays at current_l
            dist = get_dist(current_r, t, current_l)
            total_ops += dist
            current_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()