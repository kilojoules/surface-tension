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
    instructions = []
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        instructions.append((h, t))

    # State: (left_hand, right_hand)
    # Initial state: (1, 2)
    # We need to find the minimum distance to reach the target state for each instruction.
    # Since N and Q are small (100), we can use a BFS-like approach or 
    # precompute distances between states.
    
    # A state is (l, r) where l != r. Total states: N * (N-1)
    # We can use a dictionary to store the current state and total distance.
    # However, the problem says the positions are uniquely determined.
    # This means for each instruction, there is only one possible state (l, r).
    
    # Let's define a function to get the shortest distance to move one hand 
    # from start to end without passing through the other hand's position.
    def get_dist(start, end, other, n):
        # The ring is 1...N. 
        # We can move clockwise or counter-clockwise.
        # But we cannot step on 'other'.
        # This means the 'other' hand blocks one of the two paths.
        # There is only one valid path: the one that doesn't contain 'other'.
        
        # Normalize to 0...N-1
        s, e, o = start - 1, end - 1, other - 1
        
        # Path 1: s -> (s+1)%N -> ... -> e
        # Path 2: s -> (s-1)%N -> ... -> e
        # Check if 'o' is on Path 1.
        # Path 1 is the range [s, e] if s <= e, else [s, N-1] union [0, e].
        
        def is_on_path1(s, e, o, n):
            if s <= e:
                return s <= o <= e
            else:
                return o >= s or o <= e

        # If 'o' is on Path 1, we must take Path 2.
        # If 'o' is on Path 2, we must take Path 1.
        # Note: o cannot be s or e per problem statement.
        
        # Distance Path 1 (clockwise):
        dist1 = (e - s) % n
        # Distance Path 2 (counter-clockwise):
        dist2 = (s - e) % n
        
        if is_on_path1(s, e, o, n):
            return dist2
        else:
            return dist1

    # Simulation
    curr_l, curr_r = 1, 2
    total_dist = 0
    
    for h, t in instructions:
        if h == 'L':
            total_dist += get_dist(curr_l, t, curr_r, N)
            curr_l = t
        else:
            total_dist += get_dist(curr_r, t, curr_l, N)
            curr_r = t
            
    print(total_dist)

if __name__ == "__main__":
    solve()