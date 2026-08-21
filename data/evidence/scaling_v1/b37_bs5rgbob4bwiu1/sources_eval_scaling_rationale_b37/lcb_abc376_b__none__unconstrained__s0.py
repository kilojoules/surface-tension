import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T) tuples
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]

    # Helper to calculate the shortest distance between two points on a ring
    # while avoiding a specific blocked point.
    # Since we can only move one hand, the other hand acts as a wall.
    # The distance is the minimum of clockwise and counter-clockwise paths
    # that do not cross the other hand's position.
    def get_dist(start, end, block, n):
        if start == end:
            return 0
        
        # There are two paths on a ring: 
        # 1. Clockwise (increasing index)
        # 2. Counter-clockwise (decreasing index)
        
        # To check if a path is blocked, we check if 'block' lies between start and end.
        # Because we are on a ring, we can normalize coordinates.
        
        # Path A: start -> start+1 -> ... -> end (mod N)
        # Path B: start -> start-1 -> ... -> end (mod N)
        
        # We can simulate the movement or use logic:
        # A path is blocked if the block is "in the way".
        # Since we can't jump over the other hand, we must move in the direction
        # that doesn'//t encounter the block.
        
        # Let's use a simple BFS-like approach via recursion or a list comprehension
        # to find the shortest path avoiding the block, but since loops are forbidden,
        # we use a recursive helper to find the distance.
        
        def find_path(curr, target, obstacle, direction, steps):
            # direction: 1 for clockwise, -1 for counter-clockwise
            nxt = (curr + direction - 1) % N + 1
            if nxt == target:
                return steps + 1
            if nxt == obstacle:
                return float('inf')
            # To prevent infinite recursion in impossible cases (though guaranteed achievable)
            if steps > N:
                return float('inf')
            return find_path(nxt, target, obstacle, direction, steps + 1)

        return min(find_path(start, end, block, 1, 0), 
                   find_path(start, end, block, -1, 0))

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move L to t, R stays at r
            dist = get_dist(l, t, r, N)
            return (t, r, total + dist)
        else:
            # Move R to t, L stays at l
            dist = get_dist(r, t, l, N)
            return (l, t, total + dist)

    final_state = reduce(transition, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    # Increase recursion depth for the recursive distance helper
    sys.setrecursionlimit(2000)
    solve()