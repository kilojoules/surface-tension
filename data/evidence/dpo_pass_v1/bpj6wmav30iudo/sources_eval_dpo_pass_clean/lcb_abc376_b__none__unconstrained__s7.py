import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    instructions = [(input_data[i], int(input_data[i+1])) 
                    for i in range(2, len(input_data), 2)]

    # Function to calculate the shortest distance between two points on a ring of size N
    # without passing through a forbidden point 'obstacle'.
    # Since N is small (100), we can check both directions (clockwise and counter-clockwise).
    # A path is valid if the obstacle is not on the path.
    get_dist = lambda start, end, obstacle: min(
        [
            # Clockwise distance
            (end - start) % N if not any(
                (start + k) % N == (obstacle - 1) % N 
                for k in range(1, (end - start) % N + 1)
            ) else float('inf'),
            # Counter-clockwise distance
            (start - end) % N if not any(
                (start - k) % N == (obstacle - 1) % N 
                for k in range(1, (start - end) % N + 1)
            ) else float('inf')
        ]
    )

    # Note: The logic above uses 0-indexing for modulo. 
    # Let's refine the distance logic to be safer:
    # To move from 's' to 'e' avoiding 'o':
    # Path 1: s -> s+1 -> ... -> e (mod N)
    # Path 2: s -> s-1 -> ... -> e (mod N)
    
    # We use reduce to iterate through instructions. 
    # State is (current_L, current_R, total_distance)
    # Initial state: L=1, R=2, dist=0
    
    result = reduce(
        lambda state, instr: (
            # New L
            (instr[1] if instr[0] == 'L' else state[0]),
            # New R
            (instr[1] if instr[0] == 'R' else state[1]),
            # New total distance
            state[2] + (
                # If moving L, calculate dist from L to T avoiding R
                # If moving R, calculate dist from R to T avoiding L
                # Using 0-indexed logic internally: (pos-1)
                min([
                    # Clockwise
                    (instr[1] - state[0]) % N if instr[0] == 'L' and 
                    not any((state[0] + k - 1) % N + 1 == state[1] for k in range(1, (instr[1] - state[0]) % N + 1))
                    else (
                        (instr[1] - state[1]) % N if instr[0] == 'R' and 
                        not any((state[1] + k - 1) % N + 1 == state[0] for k in range(1, (instr[1] - state[1]) % N + 1))
                        else float('inf')
                    ),
                    # Counter-Clockwise
                    (state[0] - instr[1]) % N if instr[0] == 'L' and 
                    not any((state[0] - k - 1) % N + 1 == state[1] for k in range(1, (state[0] - instr[1]) % N + 1))
                    else (
                        (state[1] - instr[1]) % N if instr[0] == 'R' and 
                        not any((state[1] - k - 1) % N + 1 == state[0] for k in range(1, (state[1] - instr[1]) % N + 1))
                        else float('inf')
                    )
                ])
            )
        ),
        instructions,
        (1, 2, 0)
    )

    print(result[2])

if __name__ == "__main__":
    solve()