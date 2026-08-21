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
    total_ops = 0
    
    # Process instructions
    # The input is a list: [N, Q, H1, T1, H2, T2, ...]
    # Instructions start from index 2
    instructions = input_data[2:]
    
    # We need to find the shortest path for one hand while the other is stationary.
    # The ring is a graph where each node i is connected to (i-1) and (i+1).
    # Since we can't pass through the other hand, the ring becomes a linear path.
    
    # Helper to calculate distance between two points on a ring of size N
    # given that a certain point 'obstacle' cannot be crossed.
    def get_dist(start, end, obstacle, n):
        # We can move in two directions: clockwise and counter-clockwise.
        # One of these directions will be blocked by the obstacle.
        
        # Normalize to 0-indexed
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Check if the obstacle is "between" s and e in clockwise direction
        # Clockwise path: s -> s+1 -> ... -> e
        # The obstacle o is on the clockwise path if:
        # 1. s < o < e (standard case)
        # 2. s < o or e < o (wrap around case: s -> N-1 -> 0 -> e)
        
        # A simpler way: The obstacle splits the ring into a path of length N-1.
        # We can map the ring to a line by treating the obstacle as the boundary.
        # Let's shift all indices so that the obstacle is at N-1.
        # New index x = (x - o - 1) % N
        # Obstacle is now at N-1. The remaining nodes are 0, 1, ..., N-2.
        # The distance is simply |new_s - new_e|.
        
        ns = (s - o - 1) % N
        ne = (e - o - 1) % N
        return abs(ns - ne)

    # Iterate through instructions using a loop (since we can't use recursion)
    # We use a list comprehension to trigger the logic and then sum the results.
    
    # To maintain state (current_l, current_r) across the instructions, 
    # we use a mutable object or a reduce-like approach.
    
    state = {'l': 1, 'r': 2, 'total': 0}
    
    def process(instr_pair):
        h, t = instr_pair
        t = int(t)
        if h == 'L':
            dist = get_dist(state['l'], t, state['r'], N)
            state['total'] += dist
            state['l'] = t
        else:
            dist = get_dist(state['r'], t, state['l'], N)
            state['total'] += dist
            state['r'] = t
        return state['total']

    # Group instructions into pairs of (H, T)
    pairs = [instructions[i:i+2] for i in range(0, len(instructions), 2)]
    
    # Use map to execute the process function for each pair
    list(map(process, pairs))
    
    print(state['total'])

if __name__ == "__main__":
    solve()