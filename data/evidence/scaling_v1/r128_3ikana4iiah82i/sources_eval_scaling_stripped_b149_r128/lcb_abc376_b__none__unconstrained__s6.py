import sys
from itertools import accumulate

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (hand, target)
    # H_i is at index 2 + 2*i, T_i is at index 3 + 2*i
    instructions = [
        (input_data[2 + 2*i], int(input_data[3 + 2*i]))
        for i in range(Q)
    ]

    # Helper to calculate distance between two points on a ring of size N
    # Since we can move in both directions, the distance is the 
    # shortest path avoiding the other hand.
    # However, the problem guarantees the instructions are achievable.
    # On a ring, there are only two paths between any two points.
    # One path is clockwise, one is counter-clockwise.
    # The "blocked" path is the one that contains the other hand's current position.
    
    def get_dist(start, end, blocker, n):
        if start == end:
            return 0
        
        # Path 1: Clockwise (increasing index)
        # Distance is (end - start) % n
        # This path is blocked if the blocker is "between" start and end clockwise.
        # The blocker is at relative position (blocker - start) % n.
        # The path is blocked if 0 < (blocker - start) % n < (end - start) % n.
        
        dist_cw = (end - start) % n
        blocked_cw = (0 < (blocker - start) % n < dist_cw)
        
        # Path 2: Counter-clockwise (decreasing index)
        # Distance is (start - end) % n
        # The path is blocked if the blocker is "between" start and end counter-clockwise.
        
        dist_ccw = (start - end) % n
        blocked_ccw = (0 < (start - blocker) % n < dist_ccw)
        
        # We need the minimum distance of the non-blocked path.
        # The problem guarantees at least one path is open.
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        return min(res)

    # State: (left_hand, right_hand, total_distance)
    # Initial state: Left=1, Right=2, Dist=0
    initial_state = (1, 2, 0)
    
    # Use accumulate to iterate through instructions without a for/while loop
    # The lambda takes the current state and the next instruction
    def transition(state, instr):
        l, r, d = state
        hand, target = instr
        if hand == 'L':
            # Move left hand to target, right hand stays at r
            return (target, r, d + get_dist(l, target, r, N))
        else:
            # Move right hand to target, left hand stays at l
            return (l, target, d + get_dist(r, target, l, N))

    final_state = list(accumulate(instructions, transition, initial=initial_state))[-1]
    
    # The result is the total distance (the third element of the state tuple)
    print(final_state[2])

if __name__ == "__main__":
    solve()