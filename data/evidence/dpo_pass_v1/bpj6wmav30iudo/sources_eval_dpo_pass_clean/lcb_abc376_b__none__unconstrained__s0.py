import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Function to calculate the shortest distance between two points on a ring
    # considering the obstacle (the other hand)
    # Since N is small (100), we can evaluate both directions.
    # However, the rule is: you cannot move to the part the other hand is holding.
    # This means you can only move in the direction that doesn't cross the other hand.
    # On a ring, there are only two paths. If one is blocked by the other hand,
    # you must take the other. If neither is blocked (not possible here since 
    # hands are on the ring), you take the minimum.
    # Actually, since you can't jump over the other hand, there is only one 
    # valid path between any two points that doesn't pass through the other hand's position.
    
    get_dist = lambda start, end, obstacle: (
        # Clockwise distance
        (end - start) % N if (start < end and obstacle > start and obstacle < end) or \
                               (start > end and not (obstacle > start or obstacle < end))
        # This logic is tricky. Let's use a simpler approach:
        # There are two paths: (start -> start+1 -> ... -> end) and (start -> start-1 -> ... -> end)
        # One of these paths contains the 'obstacle' vertex. 
        # We must take the path that does NOT contain the obstacle.
        # Path 1: start, (start+1)%N, ... end
        # Path 2: start, (start-1)%N, ... end
        # Let's represent positions 1..N as 0..N-1 for easier modulo.
        0 # placeholder
    )

    # Correct logic for distance on ring without crossing obstacle:
    # Let positions be 0-indexed.
    # Path A (increasing): start -> (start+1)%N -> ... -> end
    # Path B (decreasing): start -> (start-1)%N -> ... -> end
    # Path A is blocked if obstacle is "between" start and end in increasing order.
    # Path B is blocked if obstacle is "between" start and end in decreasing order.
    
    # To avoid loops/recursion, we use a helper to determine the only viable path length.
    # Since we can't move the other hand, the "gap" between hands is the only area we can move in.
    # The distance is simply the absolute difference if the obstacle isn't in the way,
    # but since it's a ring, we just check which direction is clear.
    
    calc_move = lambda s, e, o: (
        # s, e, o are 0-indexed
        # Distance moving clockwise
        (e - s) % N if not (
            # Obstacle is in the clockwise path from s to e
            (o - s) % N < (e - s) % N
        ) else (
            # Must move counter-clockwise
            (s - e) % N
        )
    )

    # Use reduce to process instructions. 
    # State: (current_l, current_r, total_dist)
    # Initial state: L=0, R=1 (0-indexed)
    result = reduce(
        lambda state, instr: (
            # If H == 'L', move left hand to T, right hand stays
            ( (instr[1]-1), state[1], state[2] + calc_move(state[0], instr[1]-1, state[1]) )
            if instr[0] == 'L' else
            # If H == 'R', move right hand to T, left hand stays
            ( state[0], (instr[1]-1), state[2] + calc_move(state[1], instr[1]-1, state[0]) )
        ),
        instructions,
        (0, 1, 0)
    )

    print(result[2])

if __name__ == "__main__":
    solve()