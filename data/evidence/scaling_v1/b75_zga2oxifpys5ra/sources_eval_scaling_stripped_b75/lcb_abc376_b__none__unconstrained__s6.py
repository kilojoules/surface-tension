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

    # The state is (left_hand, right_hand)
    # Initial state: L=1, R=2
    # We need to find the minimum distance to move one hand to T_i 
    # without passing through the other hand's current position.
    
    # Since N and Q are small (100), we can use a simple loop.
    # The problem guarantees the instructions are achievable.
    # In a ring of size N, there are two paths between any two points.
    # One path is clockwise, one is counter-clockwise.
    # One of these paths will be blocked by the other hand.
    
    def get_dist(start, end, blocker, n):
        # There are two directions to move from start to end on a ring of size n.
        # Direction 1: Increasing indices (1 -> 2 -> ... -> n -> 1)
        # Direction 2: Decreasing indices (1 -> n -> ... -> 2 -> 1)
        
        # Check if blocker is in the path for each direction.
        # A path is blocked if the blocker is encountered before reaching the destination.
        
        # Clockwise distance and check
        # To simplify, use 0-indexed logic internally
        s, e, b = start - 1, end - 1, blocker - 1
        
        # Clockwise: s -> s+1 -> ... -> e
        # The elements in the clockwise path from s to e are:
        # (s + k) % n for k from 0 to (e - s) % n
        cw_dist = (e - s) % n
        cw_blocked = any((s + k) % n == b for k in range(cw_dist + 1))
        # Wait, the rule is "you can do this only if the other hand is not on the destination part."
        # So we check if the blocker is at any step of the movement.
        # The starting position is already occupied by the hand, and the blocker is the other hand.
        # The blocker cannot be at the start (given) or the end (given).
        # So we check if b is in {(s+k)%n for k in range(1, cw_dist)}
        
        # Correct logic for "is blocked":
        # Clockwise path: s -> (s+1)%n -> ... -> e
        # It is blocked if b is one of the intermediate steps.
        # However, the problem says "you can move... if the other hand is not on the destination part."
        # This means for every single step, the target cell must be empty.
        
        # Let's use a helper to check if a path is clear
        def is_clear(s, e, b, direction):
            # direction 1: clockwise, -1: counter-clockwise
            curr = s
            dist = 0
            while curr != e:
                curr = (curr + direction) % n
                dist += 1
                if curr == b:
                    return False, dist
            return True, dist

        res_cw = is_clear(s, e, b, 1)
        res_ccw = is_clear(s, e, b, -1)
        
        # We want the minimum distance among the clear paths.
        # Since it's guaranteed to be achievable, at least one is clear.
        options = [res_cw[1] for res_cw in [res_cw] if res_cw[0]] + \
                  [res_ccw[1] for res_ccw in [res_ccw] if res_ccw[0]]
        
        return min(options)

    # Initial positions
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