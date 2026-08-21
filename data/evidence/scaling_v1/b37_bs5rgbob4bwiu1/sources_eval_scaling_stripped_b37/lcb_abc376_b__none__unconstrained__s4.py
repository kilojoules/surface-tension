import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    instructions = []
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        instructions.append((h, t))
        idx += 2

    # State: (left_pos, right_pos)
    # We want to find the minimum operations to satisfy all instructions.
    # Since N and Q are small (100), and the problem guarantees 
    # the positions are uniquely determined, we can use DP.
    # However, the "uniquely determined" part implies a deterministic 
    # state transition. Let's use a dictionary to store {state: min_ops}.
    
    # Initial state: Left=1, Right=2
    # dp[state] = min_ops
    current_states = {(1, 2): 0}

    for h, t in instructions:
        next_states = {}
        for (l, r), cost in current_states.items():
            # We need to move hand h to position t without moving the other hand.
            # The other hand acts as a blocker.
            # Distance on a ring of size N from a to b avoiding blocker p:
            # There are two paths: clockwise and counter-clockwise.
            # A path is valid if the blocker p is not on it.
            
            # Function to calculate distance moving from 'start' to 'end' avoiding 'blocker'
            def get_dist(start, end, blocker):
                # Clockwise distance (increasing index)
                # Path: start -> start+1 -> ... -> end
                # The blocker is on the path if it's between start and end (inclusive)
                # Since we can't move the other hand, the blocker is fixed.
                # We check if the blocker is encountered.
                
                # Instead of complex logic, since N is small, we can use BFS 
                # to find the shortest path from start to end avoiding blocker.
                # But the movement is restricted to adjacent parts.
                # In a ring, there are only two directions.
                
                # Direction 1: (start + i - 1) % N + 1
                # Direction 2: (start - i - 1) % N + 1
                
                # We can use a simple BFS to find the shortest path from start to end
                # while avoiding the blocker.
                q = [(start, 0)]
                visited = {start}
                head = 0
                while head < len(q):
                    curr, d = q[head]
                    head += 1
                    if curr == end:
                        return d
                    
                    # Neighbors in ring
                    for nxt in [(curr % N) + 1, (curr - 2) % N + 1]:
                        if nxt != blocker and nxt not in visited:
                            visited.add(nxt)
                            q.append((nxt, d + 1))
                return float('inf')

            if h == 'L':
                d = get_dist(l, t, r)
                new_state = (t, r)
            else:
                d = get_dist(r, t, l)
                new_state = (l, t)
            
            if d != float('inf'):
                new_cost = cost + d
                if new_state not in next_states or new_cost < next_states[new_state]:
                    next_states[new_state] = new_cost
        
        current_states = next_states

    print(min(current_states.values()))

if __name__ == "__main__":
    solve()