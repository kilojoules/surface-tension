import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'obstacle'
    def get_dist(start, end, obstacle):
        # The ring is 1-indexed. We can think of it as 0 to N-1 for easier modulo.
        # However, we can just work with 1-N and handle the wrap-around.
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One path is blocked if the obstacle lies on it.
        
        # Path 1: start -> start+1 -> ... -> end (modulo N)
        # Path 2: start -> start-1 -> ... -> end (modulo N)
        
        # To simplify, we check if the obstacle is "between" start and end
        # in either direction.
        
        # Normalize to 0-indexed for easier range checks
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Distance going "right" (increasing index)
        dist_right = (e - s) % N
        # Check if obstacle is encountered going right
        # Obstacle is encountered if (o - s) % N < dist_right
        blocked_right = (o - s) % N < dist_right
        
        # Distance going "left" (decreasing index)
        dist_left = (s - e) % N
        # Check if obstacle is encountered going left
        # Obstacle is encountered if (s - o) % N < dist_left
        blocked_left = (s - o) % N < dist_left
        
        # We are guaranteed the instruction is achievable, 
        # so at least one path is always open.
        res = []
        if not blocked_right: res.append(dist_right)
        if not blocked_left: res.append(dist_left)
        
        return min(res)

    # State: (current_l, current_r, total_dist)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand r is the obstacle
            d = get_dist(l, t, r)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand l is the obstacle
            d = get_dist(r, t, l)
            return (l, t, total + d)

    # Use reduce to simulate the sequence of instructions
    final_state = reduce(transition, instructions, initial_state)
    
    # The result is the accumulated distance
    print(final_state[2])

if __name__ == "__main__":
    solve()