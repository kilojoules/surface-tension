import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    instructions = [ (input_data[i], int(input_data[i+1])) 
                     for i in range(2, len(input_data), 2) ]

    # Function to calculate the shortest distance between start and end on a ring of size N
    # while avoiding the position of the other hand (obstacle).
    # Since we can't use loops, we evaluate both directions (clockwise and counter-clockwise).
    # A path is valid if the obstacle is not on the path.
    # However, the problem guarantees instructions are achievable.
    # On a ring, there are only two paths. If one is blocked by the other hand, 
    # we must take the other.
    get_dist = lambda s, e, obs: (
        # Clockwise distance
        (e - s) % N if (
            # Check if obstacle is in the clockwise path
            # Obstacle is in path if (obs - s) % N < (e - s) % N
            not ((obs - s) % N < (e - s) % N)
        ) else 
        # Counter-clockwise distance
        (s - e) % N
    )

    # Use reduce to iterate through instructions and maintain the state (L, R, total_dist)
    # Initial state: Left hand at 1, Right hand at 2, total distance 0
    final_state = reduce(
        lambda state, inst: (
            # If instruction is for Left hand (L)
            (inst[1], state[1], state[2] + get_dist(state[0], inst[1], state[1]))
            if inst[0] == 'L' else
            # If instruction is for Right hand (R)
            (state[0], inst[1], state[2] + get_dist(state[1], inst[1], state[0]))
        ),
        instructions,
        (1, 2, 0)
    )

    # The result is the accumulated distance in the third element of the state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()