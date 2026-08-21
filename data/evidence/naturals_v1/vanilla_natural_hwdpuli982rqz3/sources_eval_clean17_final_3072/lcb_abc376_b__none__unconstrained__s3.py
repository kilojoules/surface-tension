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
    L, R = 1, 2
    total_ops = 0
    
    # Helper to calculate shortest distance between two points on a ring
    # without passing through a forbidden point
    def get_dist(start, end, forbidden):
        if start == end:
            return 0
        
        # BFS to find shortest path on the ring (N nodes)
        # Since N is small (100), BFS is efficient
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            curr, dist = queue.popleft()
            
            # Neighbors on the ring
            # Part i is connected to (i-1) and (i+1)
            # Handle wrap-around with 1-based indexing
            neighbors = [
                (curr - 2) % N + 1, 
                curr % N + 1
            ]
            
            for neighbor in neighbors:
                if neighbor == end:
                    return dist + 1
                if neighbor != forbidden and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return float('inf')

    # Process instructions
    # The input data is [N, Q, H1, T1, H2, T2, ...]
    # Instructions start at index 2
    for i in range(Q):
        H = input_data[2 + 2*i]
        T = int(input_data[3 + 2*i])
        
        if H == 'L':
            # Move Left hand to T, Right hand is forbidden
            dist = get_dist(L, T, R)
            total_ops += dist
            L = T
        else:
            # Move Right hand to T, Left hand is forbidden
            dist = get_dist(R, T, L)
            total_ops += dist
            R = T
            
    print(total_ops)

if __name__ == "__main__":
    solve()