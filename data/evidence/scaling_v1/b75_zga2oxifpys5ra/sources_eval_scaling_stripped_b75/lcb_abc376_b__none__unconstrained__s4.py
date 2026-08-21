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
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]
    
    # Initial state: Left hand at 1, Right hand at 2
    # State is (L, R)
    # We use a list of possible states for each step.
    # Since the problem guarantees positions are uniquely determined, 
    # we can just track the current (L, R).
    
    def get_dist(start, end, obstacle, n):
        # Calculate distance from start to end on a ring of size n
        # without passing through the obstacle.
        # There are two directions: clockwise and counter-clockwise.
        
        # Clockwise distance
        # To check if obstacle is in the way:
        # The path is start -> start+1 -> ... -> end (mod n)
        # We check if the obstacle lies in the range [start, end] circularly.
        
        def is_between(s, e, obs, n):
            # Check if obs is on the path from s to e moving +1
            if s <= e:
                return s < obs <= e
            else: # Wrap around
                return obs > s or obs <= e

        # Distance clockwise
        dist_cw = (end - start) % n
        # Distance counter-clockwise
        dist_ccw = (start - end) % n
        
        # Check if obstacle blocks clockwise
        # The obstacle is at 'obstacle'. The path is s, s+1... e.
        # The obstacle blocks if it's any of the steps taken.
        # Actually, the rule is: cannot move to destination if other hand is there.
        # So we check if the obstacle is encountered during the transition.
        
        # For a move to be valid, the obstacle must not be on the path.
        # Clockwise path: start+1, start+2 ... end
        # Counter-clockwise path: start-1, start-2 ... end
        
        # We can use a helper to check if the obstacle is hit.
        # A simpler way: the obstacle divides the ring into a linear path.
        # The distance is simply the distance on that linear path.
        # If we treat the obstacle as a boundary, the distance is:
        # If we normalize coordinates such that obstacle is at 0:
        s_norm = (start - obstacle) % n
        e_norm = (end - obstacle) % n
        # Now we are on a line from 0 to N, but we can't pass 0.
        # The distance is |s_norm - e_norm|
        return abs(s_norm - e_norm)

    # Initial positions
    curr_l, curr_r = 1, 2
    total_ops = 0
    
    for h, t in instructions:
        if h == 'L':
            # Move left hand to t, right hand stays at curr_r
            total_ops += get_dist(curr_l, t, curr_r, N)
            curr_l = t
        else:
            # Move right hand to t, left hand stays at curr_l
            total_ops += get_dist(curr_r, t, curr_l, N)
            curr_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()