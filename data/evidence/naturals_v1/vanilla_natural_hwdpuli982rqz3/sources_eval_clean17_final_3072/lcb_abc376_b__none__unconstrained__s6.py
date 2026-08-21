import sys
from collections import deque

def solve():
    # Read input
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
        
        # BFS to find shortest path avoiding the forbidden part
        # Since N is small (100), BFS is efficient
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            curr, dist = queue.popleft()
            
            # Possible moves on the ring
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
    instructions = input_data[2:]
    total_ops = 0
    
    # We use a loop with an index to track current_l and current_r
    # since we cannot use for-each with state updates easily without a helper
    for i in range(0, len(instructions), 2):
        h = instructions[i]
        t = int(instructions[i+1])
        
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