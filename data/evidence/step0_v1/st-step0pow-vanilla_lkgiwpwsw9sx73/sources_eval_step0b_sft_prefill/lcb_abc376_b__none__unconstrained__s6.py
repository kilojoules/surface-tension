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
    cur_l = 1
    cur_r = 2
    
    # Helper to calculate shortest distance between two nodes on a ring of size N
    # without passing through a forbidden node 'block'
    def get_dist(start, end, block):
        # The ring is 1-indexed. We can treat it as 0-indexed for easier math.
        s, e, b = start - 1, end - 1, block - 1
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One path is blocked if 'b' is between 's' and 'e'.
        
        # To simplify, we can use BFS since N is very small (100)
        # BFS finds the shortest path in an unweighted graph.
        queue = deque([(s, 0)])
        visited = {s, b}
        
        while queue:
            curr, dist = queue.popleft()
            if curr == e:
                return dist
            
            # Neighbors on the ring
            for neighbor in [(curr + 1) % N, (curr - 1) % N]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    # Process instructions
    instructions = input_data[2:]
    total_ops = 0
    
    # We use a list to keep track of current positions because we can't use 
    # nonlocal/global easily inside a map/reduce structure.
    state = [cur_l, cur_r]
    
    def process_instruction(instr_pair):
        nonlocal total_ops
        h, t = instr_pair
        t = int(t)
        l, r = state
        
        if h == 'L':
            # Move left hand to t, right hand r is the block
            dist = get_dist(l, t, r)
            total_ops += dist
            state[0] = t
        else:
            # Move right hand to t, left hand l is the block
            dist = get_dist(r, t, l)
            total_ops += dist
            state[1] = t

    # Chunk the instructions into pairs of (H, T)
    pairs = [instructions[i:i+2] for i in range(0, len(instructions), 2)]
    
    # Use a loop to process instructions sequentially
    for p in pairs:
        process_instruction(p)
        
    print(total_ops)

if __name__ == "__main__":
    solve()