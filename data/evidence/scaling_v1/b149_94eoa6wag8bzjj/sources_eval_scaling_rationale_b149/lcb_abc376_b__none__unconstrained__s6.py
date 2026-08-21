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
    # The distance is the number of steps in the direction that doesn't hit 'obs'.
    def get_dist(start, end, obs, N):
        if start == end:
            return 0
        
        # There are two directions: clockwise and counter-clockwise.
        # One direction is blocked by the obstacle.
        # We check if the obstacle is "between" start and end in clockwise direction.
        # Clockwise distance from start to end:
        dist_cw = (end - start + N) % N
        # Clockwise distance from start to obstacle:
        dist_obs_cw = (obs - start + N) % N
        
        # If the obstacle is reached before the target in clockwise direction,
        # we must go counter-clockwise.
        # Otherwise, we can go clockwise.
        # Note: The problem guarantees the instruction is achievable.
        # If dist_obs_cw < dist_cw, the clockwise path is blocked.
        # Otherwise, the clockwise path is clear.
        
        # However, the obstacle could be at the target's position in the other direction.
        # The only way to move is the direction where the obstacle is NOT encountered.
        # Let's check if the clockwise path is clear:
        # The clockwise path is clear if the obstacle is not in the range (start, end]
        # in clockwise terms.
        
        # A simpler check: is the obstacle 'between' start and end clockwise?
        # The obstacle is between start and end clockwise if:
        # (obs - start + N) % N < (end - start + N) % N
        
        if (obs - start + N) % N < (end - start + N) % N:
            # Clockwise is blocked, must go counter-clockwise
            return (start - end + N) % N
        else:
            # Clockwise is clear
            return (end - start + N) % N

    # State: (current_l, current_r, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def step(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand stays at r
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            # Move right hand to t, left hand stays at l
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(step, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()