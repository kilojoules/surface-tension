import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H_i, T_i)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Helper to calculate distance between start and end on a ring of size N
    # given that the 'obstacle' position cannot be stepped on.
    # Since we can only move one hand, and the other is stationary,
    # we can only move in the direction that doesn't hit the obstacle.
    def get_dist(start, end, obstacle, n):
        if start == end:
            return 0
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One path is (start + 1, start + 2 ... end)
        # The other is (start - 1, start - 2 ... end)
        
        # Calculate clockwise distance
        # (end - start) mod n
        cw_dist = (end - start) % n
        # The path is start -> (start + 1)%n -> ... -> end
        # It is blocked if the obstacle is at any of these positions.
        # The obstacle is on the CW path if (obstacle - start)%n < cw_dist
        # Note: obstacle is guaranteed != end and != start (per problem statement)
        cw_blocked = (obstacle - start) % n < cw_dist
        
        # Calculate counter-clockwise distance
        ccw_dist = (start - end) % n
        # The obstacle is on the CCW path if (start - obstacle)%n < ccw_dist
        ccw_blocked = (start - obstacle) % n < ccw_dist
        
        # We are guaranteed the instruction is achievable.
        # Return the distance of the path that is not blocked.
        return cw_dist if not cw_blocked else ccw_dist

    # State: (left_pos, right_pos, total_distance)
    # Initial state: Left at 1, Right at 2
    initial_state = (1, 2, 0)
    
    # Use reduce to process instructions and accumulate distance
    final_state = reduce(
        lambda state, inst: (
            # New Left Position
            inst[1] if inst[0] == 'L' else state[0],
            # New Right Position
            inst[1] if inst[0] == 'R' else state[1],
            # New Total Distance
            state[2] + (
                get_dist(state[0], inst[1], state[1], N) if inst[0] == 'L' 
                else get_dist(state[1], inst[1], state[0], N)
            )
        ),
        instructions,
        initial_state
    )
    
    print(final_state[2])

if __name__ == "__main__":
    solve()