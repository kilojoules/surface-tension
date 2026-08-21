import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions
    # instructions = [(H_i, T_i), ...]
    instructions = [
        (input_data[2 + 2*i], int(input_data[3 + 2*i]))
        for i in range(Q)
    ]

    # Initial state: Left hand at 1, Right hand at 2
    # We use a list of tuples [(l, r)] to track possible current positions.
    # Although the problem says positions are uniquely determined, 
    # we can use a functional approach to transition states.
    
    def get_dist(start, end, obstacle, n):
        """
        Calculate the shortest distance from start to end on a ring of size n,
        without passing through the obstacle.
        Since we can only move one hand, the other hand acts as a wall.
        On a ring, there are only two directions. One will be blocked by the obstacle.
        """
        # Clockwise distance (increasing index)
        # To go from start to end clockwise:
        # If we don't cross the obstacle, the distance is (end - start) % n
        # But we must check if the obstacle is in the path.
        
        # Simplified: In a ring of size N, if one point is blocked, 
        # it becomes a linear path of N-1 nodes.
        # The distance is simply the distance in the linear path.
        # Let's normalize coordinates so the obstacle is at 'virtual' position 0.
        # The available nodes are 1, 2, ..., N-1.
        
        # Shift everything so obstacle is at N
        s = (start - obstacle) % N
        e = (end - obstacle) % N
        # Now we are on a line from 0 to N-1, and cannot pass through 0/N.
        # The distance is simply abs(s - e)
        # Wait, the shift above: if start is 1 and obstacle is 2, s = -1 % N = N-1.
        # If end is 3, e = 1 % N = 1. Distance is abs(N-1 - 1) = N-2.
        # Let's refine:
        s = (start - obstacle) % N
        if s == 0: s = N # Normalize to 1...N
        e = (end - obstacle) % N
        if e == 0: e = N
        # Since the obstacle is at '0', and we can't pass it, 
        # we are on a line from 1 to N.
        # However, the obstacle is at a specific position.
        # Let's use the property: distance is abs(s - e) where s, e are 
        # positions relative to the obstacle.
        
        # Correct logic for distance on ring with one blocked node:
        # The blocked node splits the ring into a line of length N-1.
        # The distance is the absolute difference of their positions 
        # when the ring is 'cut' at the obstacle.
        
        # Relative positions:
        rel_start = (start - obstacle) % N
        rel_end = (end - obstacle) % N
        return abs(rel_start - rel_end)

    # We use a reduction to process instructions and accumulate distance.
    # state: (current_l, current_r, total_dist)
    def process_instruction(state, instr):
        l, r, dist = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r is obstacle
            d = get_dist(l, t, r, N)
            return (t, r, dist + d)
        else:
            # Move right hand to t, left hand l is obstacle
            d = get_dist(r, t, l, N)
            return (l, t, dist + d)

    # Using a loop to simulate the reduction
    final_state = (1, 2, 0)
    for instr in instructions:
        final_state = process_instruction(final_state, instr)
    
    print(final_state[2])

if __name__ == "__main__":
    solve()