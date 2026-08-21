import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between start and end on a ring of size N
    # given that the other hand is at position 'obs'.
    # Since we cannot pass through 'obs', there is only one valid path.
    # The distance is the absolute difference if the path doesn't cross 'obs'.
    # More simply: the distance is the number of steps in the direction 
    # that does not contain 'obs'.
    def get_dist(start, end, obs, n):
        # There are two directions: clockwise and counter-clockwise.
        # One direction is blocked by 'obs'.
        # We check if 'obs' is "between" start and end in one direction.
        # A simpler way: the only available path is the one that doesn't 
        # contain 'obs'. 
        # Let's normalize coordinates to 0...N-1
        s, e, o = start - 1, end - 1, obs - 1
        
        # Distance moving clockwise (increasing index)
        # The path is s -> s+1 -> ... -> e
        # It is blocked if o is in {s+1, ..., e} (modulo N)
        # Instead of loops, we can check if the clockwise distance 
        # from s to e covers o.
        dist_cw = (e - s) % n
        # The clockwise path covers (dist_cw + 1) nodes.
        # The obstacle o is on the clockwise path if (o - s) % n < dist_cw.
        # Wait, the rule is: cannot move to the part the other hand is holding.
        # So if we move s -> e clockwise, we are blocked if o is any of the 
        # intermediate steps or the destination.
        # But the problem guarantees T_i != other_hand.
        # So we are blocked if o is strictly between s and e clockwise.
        
        is_blocked_cw = (o - s) % n < dist_cw and o != s
        
        if not is_blocked_cw:
            return dist_cw
        else:
            # Must go counter-clockwise
            return (s - e) % n

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def transition(state, instr):
        l, r, d = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand stays at r
            cost = get_dist(l, t, r, N)
            return (t, r, d + cost)
        else:
            # Move right hand to t, left hand stays at l
            cost = get_dist(r, t, l, N)
            return (l, t, d + cost)

    # Use reduce to simulate the process across all instructions
    final_state = reduce(transition, instructions, initial_state)
    
    # The result is the accumulated distance
    print(final_state[2])

if __name__ == "__main__":
    solve()