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
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]

    # State: (left_hand_pos, right_hand_pos)
    # Initial state: L=1, R=2
    # We want to find the minimum operations to satisfy Q instructions.
    # Since N and Q are small (100), we can use dynamic programming.
    # dp[i][l][r] = min operations after i instructions with hands at (l, r).
    # However, the problem guarantees that positions are uniquely determined.
    # Let's verify if we can just track the current (l, r) and calculate the 
    # shortest path for the moving hand without crossing the stationary hand.
    
    # Function to calculate shortest distance from start to end on a ring of size N
    # without passing through the obstacle.
    # The ring is 1-indexed.
    def get_dist(start, end, obstacle, N):
        # We can move clockwise or counter-clockwise.
        # A path is valid if the obstacle is not on it.
        # Since we can't jump over the obstacle, there is only one valid path
        # if the obstacle is not at the start or end.
        
        # Normalize to 0-indexed for easier modulo arithmetic
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Clockwise distance (increasing index)
        # Path: s -> s+1 -> ... -> e
        # The obstacle o is on this path if (s < o <= e) or (s > e and (o > s or o <= e))
        # Actually, simpler: the obstacle divides the ring into two arcs.
        # One arc is (o+1) % N ... (o-1) % N.
        # The distance is simply the distance along the arc that doesn't contain o.
        
        # Distance moving "right" (s -> s+1)
        # The number of steps is (e - s) % N.
        # This path is blocked if the obstacle o is encountered.
        # The obstacle is encountered if (o - s) % N < (e - s) % N.
        
        dist_cw = (e - s) % N
        blocked_cw = (o - s) % N < dist_cw
        
        # Distance moving "left" (s -> s-1)
        # The number of steps is (s - e) % N.
        # The obstacle is encountered if (s - o) % N < (s - e) % N.
        
        dist_ccw = (s - e) % N
        blocked_ccw = (s - o) % N < dist_ccw
        
        # It is guaranteed that the instruction is achievable.
        # So one of these must be False.
        if not blocked_cw:
            return dist_cw
        if not blocked_ccw:
            return dist_ccw
        return float('inf')

    # Initial positions
    curr_l, curr_r = 1, 2
    total_ops = 0
    
    for h, t in instructions:
        if h == 'L':
            # Move left hand to t, right hand stays at curr_r
            dist = get_dist(curr_l, t, curr_r, N)
            total_ops += dist
            curr_l = t
        else:
            # Move right hand to t, left hand stays at curr_l
            dist = get_dist(curr_r, t, curr_l, N)
            total_ops += dist
            curr_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()