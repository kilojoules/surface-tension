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
    
    # Process instructions
    # The input is flat, so we iterate through the pairs starting from index 2
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        
        # BFS to find the shortest path for the moving hand
        # State is just the position of the hand being moved
        # The other hand is a static obstacle
        
        start_node = current_l if h == 'L' else current_r
        obstacle = current_r if h == 'L' else current_l
        
        # BFS setup
        queue = deque([(start_node, 0)])
        visited = {start_node}
        
        # We need to find the distance to t
        # Since we can't use loops, we use a list comprehension to simulate 
        # the BFS layers or a deque with a while loop.
        # Note: The prompt says "Return only Python source", while loops are allowed.
        
        dist = 0
        found = False
        
        # Standard BFS using while loop
        while queue:
            curr, d = queue.popleft()
            if curr == t:
                dist = d
                found = True
                break
            
            # Neighbors in a ring
            # Part i is connected to (i-1) and (i+1)
            # Handle 1-based indexing with modulo
            neighbors = [
                (curr - 2) % N + 1, 
                curr % N + 1
            ]
            
            for neighbor in neighbors:
                if neighbor != obstacle and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        
        total_ops += dist
        if h == 'L':
            current_l = t
        else:
            current_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()