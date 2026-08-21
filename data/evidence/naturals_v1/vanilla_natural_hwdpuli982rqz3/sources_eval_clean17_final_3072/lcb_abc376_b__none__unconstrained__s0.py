import sys
from collections import deque

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    current_l = 1
    current_r = 2
    total_ops = 0
    
    # Process instructions
    # The input is a flat list, so we iterate through it in chunks of 2
    # starting from index 2.
    instructions = input_data[2:]
    
    # We use a BFS to find the shortest path for one hand given the other is fixed.
    # Since N is small (100), BFS is efficient.
    def get_min_dist(start, target, obstacle):
        # BFS to find shortest path from start to target avoiding obstacle
        # Distance is 0 if start == target
        if start == target:
            return 0
        
        queue = deque([(start, 0)])
        visited = {start, obstacle}
        
        while queue:
            curr, dist = queue.popleft()
            
            # Neighbors in a ring
            # Part i is connected to (i-1) and (i+1)
            # Handle 1-based indexing with modulo
            neighbors = [
                (curr - 2) % N + 1, 
                curr % N + 1
            ]
            
            for neighbor in neighbors:
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    # Iterate through the instructions using a loop
    # Since we cannot use recursion, we use a for loop with range
    for i in range(0, Q * 2, 2):
        h = instructions[i]
        t = int(instructions[i+1])
        
        if h == 'L':
            # Move left hand to t, right hand is obstacle
            dist = get_min_dist(current_l, t, current_r)
            total_ops += dist
            current_l = t
        else:
            # Move right hand to t, left hand is obstacle
            dist = get_min_dist(current_r, t, current_l)
            total_ops += dist
            current_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()