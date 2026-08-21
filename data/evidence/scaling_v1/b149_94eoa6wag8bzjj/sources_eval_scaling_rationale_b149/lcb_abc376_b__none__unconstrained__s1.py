import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate the shortest distance between start and end on a ring of size N,
    # given that the other hand (obstacle) is at position 'obs'.
    # Since we cannot pass through the obstacle, there is only one valid direction.
    def get_dist(start, end, obs, n):
        # The ring is 1-indexed.
        # We can move clockwise or counter-clockwise.
        # One direction is blocked by the obstacle.
        
        # Check if the obstacle is "between" start and end in clockwise direction.
        # Clockwise distance from start to end:
        cw_dist = (end - start + n) % n
        # Clockwise distance from start to obs:
        s_to_obs_cw = (obs - start + n) % n
        
        # If the obstacle is reached before the target in clockwise direction,
        # we MUST go counter-clockwise.
        # Note: start != obs and end != obs is guaranteed.
        # If s_to_obs_cw < cw_dist, the clockwise path is blocked.
        # Otherwise, the clockwise path is clear.
        # However, the problem says we can't move to the destination part if the other hand is there.
        # The only way to be blocked is if the obstacle lies on the path.
        
        # Let's evaluate both directions:
        # 1. Clockwise: start -> start+1 -> ... -> end
        # Path is blocked if obs is any of the steps.
        # 2. Counter-clockwise: start -> start-1 -> ... -> end
        # Path is blocked if obs is any of the steps.
        
        # A path is blocked if the obstacle is in the arc between start and end.
        # Clockwise arc: { (start + k) % N for k in 1..cw_dist }
        # The obstacle is in the clockwise arc if (obs - start + n) % n <= cw_dist
        # Wait, the condition is: we cannot move to the destination part if the other hand is there.
        # But the guarantee says T_i != other_hand.
        # So we just need to check if the obstacle is strictly between start and end.
        
        # Clockwise distance
        d_cw = (end - start + n) % n
        # Counter-clockwise distance
        d_ccw = (start - end + n) % n
        
        # The obstacle blocks the clockwise path if it's encountered.
        # The obstacle is at 'obs'. The clockwise path is blocked if 
        # (obs - start + n) % n < d_cw (since obs != end)
        # Actually, the simplest check: is the obstacle in the clockwise arc?
        # The clockwise arc is blocked if (obs - start + n) % n < d_cw.
        # But we must check if the obstacle is "between" them.
        # Since we can't jump over the other hand, only one arc is available.
        
        # If we move clockwise, we visit (start + k) % N.
        # If (obs - start + n) % n < d_cw, clockwise is blocked.
        # Otherwise, counter-clockwise is blocked (or both are clear if N=2, but N>=3).
        
        # Let's refine:
        # Clockwise distance is d_cw. The obstacle is at distance (obs-start+n)%n.
        # If (obs-start+n)%n < d_cw, the clockwise path is blocked.
        # Otherwise, the counter-clockwise path is blocked if (start-obs+n)%n < d_ccw.
        
        # Because the problem guarantees the move is possible, one path must be open.
        # If (obs - start + n) % n < d_cw, we must go CCW.
        # Else, we can go CW (which is shorter or equal if we choose greedily).
        # Wait, the problem asks for the MINIMUM total operations.
        # But we can't pass through the obstacle. So we don't have a choice of direction
        # if one is blocked. If both are open (not possible on a ring with 2 hands), 
        # we'd take the min. With one obstacle, only one arc is available.
        
        # Actually, the only way both arcs are open is if the obstacle is not on either,
        # which is impossible. One arc always contains the obstacle.
        # The only exception is if the obstacle is the start or end, but that's forbidden.
        
        # Correct logic:
        # The obstacle 'obs' divides the ring into one path.
        # The distance is simply the distance from start to end going the "long way" 
        # around the obstacle.
        # The available path is the one that does NOT contain 'obs'.
        # The clockwise path is start -> start+1 -> ... -> end.
        # It is blocked if (obs - start + n) % n < d_cw.
        # If blocked, distance is d_ccw. Otherwise, distance is d_cw.
        # Wait, if (obs - start + n) % n > d_cw, the clockwise path is clear.
        # If (obs - start + n) % n < d_cw, the clockwise path is blocked.
        # What if (obs - start + n) % n == d_cw? That means obs == end, forbidden.
        
        return d_ccw if (obs - start + n) % n < d_cw else d_cw

    # State: (left_hand, right_hand, total_dist)
    # Initial state: L=1, R=2, dist=0
    def step(state, instr):
        l, r, d = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand r is obstacle
            # Use 0-indexing for modulo logic
            dist = get_dist(l-1, t-1, r-1, N)
            return (t, r, d + dist)
        else:
            # Move right hand to t, left hand l is obstacle
            dist = get_dist(r-1, t-1, l-1, N)
            return (l, t, d + dist)

    final_state = reduce(step, instructions, (1, 2, 0))
    print(final_state[2])

if __name__ == "__main__":
    solve()