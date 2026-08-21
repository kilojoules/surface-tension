import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    # instructions = [('R', 4), ('L', 5), ...]
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Function to calculate the shortest distance between start and end on a ring
    # while avoiding the obstacle (the other hand).
    # Since N is small (100), we can check both directions.
    # Clockwise distance: (end - start) % N
    # Counter-clockwise distance: (start - end) % N
    # However, we must check if the obstacle is in the way.
    # On a ring, the only way to get from A to B is blocked by C 
    # if C is on the path. 
    # Because we can't pass through the other hand, we only have one 
    # viable direction if the other hand is "between" the start and end.
    
    # Let's define a helper to check if the obstacle is in the path 
    # from 's' to 'e' moving in a specific direction.
    # For a ring of size N, moving from s to e:
    # Path 1 (Increasing): s, s+1, ..., e (modulo N)
    # Path 2 (Decreasing): s, s-1, ..., e (modulo N)
    
    # Since we need to avoid loops/recursion, we use a mathematical 
    # approach to determine the distance.
    # The distance is simply the absolute difference if the obstacle 
    # is not in the way.
    # Actually, on a ring, if you can't pass the other hand, 
    # you are essentially on a line of length N-1.
    # The relative position of the hands is fixed.
    
    # Let's track the state as (left_pos, right_pos, total_dist)
    # Initial state: L=1, R=2
    initial_state = (1, 2, 0)

    def calculate_dist(s, e, obs, n):
        # Normalize to 0-indexed for easier modulo math
        s, e, obs = s-1, e-1, obs-1
        
        # Distance moving clockwise
        # The path is s, (s+1)%n, (s+2)%n ... e
        # The number of steps is (e - s) % n
        # The obstacle is hit if (obs - s) % n < (e - s) % n
        dist_cw = (e - s) % n
        blocked_cw = (obs - s) % n < dist_cw
        
        # Distance moving counter-clockwise
        # The path is s, (s-1)%n, (s-2)%n ... e
        # The number of steps is (s - e) % n
        # The obstacle is hit if (s - obs) % n < (s - e) % n
        dist_ccw = (s - e) % n
        blocked_ccw = (s - obs) % n < dist_ccw
        
        # Return the minimum distance of the non-blocked path
        # It is guaranteed that at least one path is open.
        return min(
            dist_cw if not blocked_cw else float('inf'),
            dist_ccw if not blocked_ccw else float('inf')
        )

    def process_instruction(state, instr):
        l, r, total = state
        h, t = instr
        
        if h == 'L':
            # Move left hand to t, right hand stays at r
            d = calculate_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand stays at l
            d = calculate_dist(r, t, l, N)
            return (l, t, total + d)

    # Use reduce to iterate through instructions without a for-loop
    final_state = reduce(process_instruction, instructions, initial_state)
    
    # Output the total distance
    sys.stdout.write(str(int(final_state[2])) + '\n')

if __name__ == "__main__":
    solve()