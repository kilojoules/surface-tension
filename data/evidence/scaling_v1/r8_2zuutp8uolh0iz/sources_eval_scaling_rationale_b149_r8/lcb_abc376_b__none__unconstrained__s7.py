import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as pairs of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between start and end on a ring of size N
    # while avoiding the position of the other hand 'obs'.
    # Since the other hand is stationary, the ring becomes a line.
    # The distance is simply the absolute difference if we "cut" the ring at 'obs'.
    def get_dist(start, end, obs, n):
        # Normalize coordinates so that obs is at position N+1 (outside the range)
        # We shift everything so obs is the boundary.
        # New coordinate x' = (x - obs - 1) % n
        s = (start - obs - 1) % n
        e = (end - obs - 1) % n
        # In the linear space [0, n-1], the distance is just |s - e|
        # because we cannot cross the 'obs' point.
        return abs(s - e)

    # State: (current_l, current_r, total_dist)
    # reduce(function, sequence, initial)
    final_state = reduce(
        lambda state, inst: (
            # New state: (new_l, new_r, new_total_dist)
            (
                inst[1] if inst[0] == 'L' else state[0], # new_l
                inst[1] if inst[0] == 'R' else state[1], # new_r
                state[2] + (
                    get_dist(state[0], inst[1], state[1], N) if inst[0] == 'L' 
                    else get_dist(state[1], inst[1], state[0], N)
                )
            )
        ),
        instructions,
        (1, 2, 0) # Initial: left=1, right=2, dist=0
    )

    print(final_state[2])

if __name__ == "__main__":
    solve()