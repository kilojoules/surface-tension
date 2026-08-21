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

    # Initial state: Left hand at 1, Right hand at 2
    # State is (left, right)
    # We want to find the minimum total distance.
    # Since N and Q are small (100), we can use DP.
    # dp[i][l][r] = min operations to complete i instructions ending at (l, r)
    # However, the problem says the positions are uniquely determined.
    # Let's verify if we can just simulate it.
    # For a given instruction (H, T), only one hand moves.
    # The other hand acts as a blocker.
    # On a ring of size N, there are two paths between any two points.
    # One path is blocked if the other hand is on it.
    
    def get_dist(start, end, blocker, n):
        # The hand moves from start to end on a ring of size n.
        # The blocker is at position 'blocker'.
        # There are two directions: clockwise and counter-clockwise.
        # A path is valid if the blocker is not visited.
        
        # Clockwise distance (increasing index)
        # Path: start -> start+1 -> ... -> end
        # The blocker is encountered if it lies between start and end (exclusive of start, inclusive of end)
        # But the rule is: cannot move to destination if other hand is there.
        # So we check if the blocker is in the set of nodes visited.
        
        # Since we need the minimum operations and only one hand moves,
        # we just need to check which of the two arcs is clear of the blocker.
        
        # Arc 1: start -> start + 1 -> ... -> end (mod N)
        # Arc 2: start -> start - 1 -> ... -> end (mod N)
        
        # To simplify, we can use a helper to check if a value is in the range [a, b] modulo N.
        def is_between(a, b, x, n):
            # Check if x is in the path from a to b moving clockwise
            # Normalize to 0..N-1
            a, b, x = (a-1)%n, (b-1)%n, (x-1)%n
            if a <= b:
                return a < x <= b
            else: # wrap around
                return x > a or x <= b

        # Distance clockwise
        dist_cw = (end - start) % N
        # Distance counter-clockwise
        dist_ccw = (start - end) % N
        
        # Check if blocker is in the clockwise path
        # The blocker blocks the move if it is any of the nodes in the path
        # from start+1 to end.
        blocked_cw = is_between(start, end, blocker, N)
        blocked_ccw = is_between(end, start, blocker, N)
        
        # It is guaranteed that the instruction is achievable.
        # So at least one path is always clear.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        
        return min(res)

    # Simulation
    curr_l, curr_r = 1, 2
    total_ops = 0
    
    for h, t in instructions:
        if h == 'L':
            total_ops += get_dist(curr_l, t, curr_r, N)
            curr_l = t
        else:
            total_ops += get_dist(curr_r, t, curr_l, N)
            curr_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()