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

    # State: (left_hand, right_hand)
    # Initial state: (1, 2)
    # We want to find the minimum operations to satisfy Q instructions.
    # Since N and Q are small (100), and the other hand must stay fixed,
    # the movement is simply the shortest path on a ring avoiding the other hand.
    
    # A function to calculate the distance from start to end on a ring of size N,
    # avoiding a forbidden position 'obstacle'.
    # Since we can't jump over the obstacle, there is only one path (clockwise or counter-clockwise).
    def get_dist(start, end, obstacle, n):
        if start == end:
            return 0
        
        # The ring is 1...N. We can move in two directions.
        # One direction is blocked by the obstacle.
        # We check which direction is clear.
        
        # Function to check if a path from a to b (moving +1 mod N) is clear
        def is_clear(a, b, obs, n):
            # Path: a, a+1, ..., b (all mod N)
            # We check if obs is in the sequence of nodes visited.
            # The destination b is allowed, but the path to it must not hit obs.
            # Actually, the rule is: "you can do this only if the other hand is not on the destination part."
            # So we just need to check if the obstacle is encountered during the transition.
            
            # Using a simple loop to simulate movement:
            curr = a
            steps = 0
            while curr != b:
                curr = (curr % n) + 1
                if curr == obs:
                    return False
                steps += 1
            return True

        # Try clockwise
        # We use a helper to calculate distance if clear
        def dist_cw(a, b, n):
            return (b - a) % n if (b - a) % n != 0 else 0
        
        # Since we can't pass the obstacle, only one direction is possible.
        # We check if the clockwise path is clear.
        # Clockwise path from s to e: s -> s+1 -> ... -> e
        # It is clear if the obstacle 'o' is not in {s+1, ..., e}.
        
        # To simplify: the obstacle divides the ring into a linear path.
        # The distance is simply the distance on that linear path.
        # Normalize coordinates to 0...N-1 and shift so obstacle is at the boundary.
        # New coordinate x' = (x - obstacle - 1) % N
        # This maps the obstacle to N-1 and the remaining parts to 0...N-2.
        s_norm = (start - obstacle - 1) % N
        e_norm = (end - obstacle - 1) % N
        # Now we are on a line from 0 to N-2. The distance is simply |s_norm - e_norm|.
        return abs(s_norm - e_norm)

    # Initial positions
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