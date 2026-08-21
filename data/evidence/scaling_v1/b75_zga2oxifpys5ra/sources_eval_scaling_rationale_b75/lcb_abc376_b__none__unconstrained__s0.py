import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions parsed into a list of (H, T)
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]

    # Function to calculate the shortest distance between start and end on a ring of size N,
    # given that the other hand is at position 'obstacle'.
    # The movement must not pass through the obstacle.
    def get_dist(start, end, obstacle, n):
        # There are two directions to move on a ring: clockwise and counter-clockwise.
        # We check if the obstacle lies on the path for each direction.
        
        # Clockwise distance and path check
        # Path: start -> start+1 -> ... -> end
        # The obstacle is encountered if it lies 'between' start and end in clockwise direction.
        # Simplified: if we move clockwise, we hit the obstacle if 
        # (obstacle - start) % n < (end - start) % n
        
        cw_dist = (end - start) % n
        ccw_dist = (start - end) % n
        
        # Check if obstacle is in the way for clockwise movement
        # The obstacle is in the way if (obstacle - start) % n is less than cw_dist
        # AND the obstacle is not the start point (though problem says T != obstacle)
        cw_blocked = (obstacle - start) % n < cw_dist
        ccw_blocked = (start - obstacle) % n < ccw_dist
        
        # We are guaranteed the instruction is achievable, so at least one path is open.
        # If clockwise is blocked, we must go counter-clockwise, and vice versa.
        # If neither is blocked (impossible for N > 2 and obstacle present), we take min.
        # If both are blocked (impossible per guarantee), this logic still holds.
        
        res = []
        if not cw_blocked: res.append(cw_dist)
        if not ccw_blocked: res.append(ccw_dist)
        
        return min(res) if res else 0

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    # Use reduce to iterate through instructions and update the state
    final_state = reduce(
        lambda state, instr: (
            # New State: (new_l, new_r, new_dist)
            (
                instr[1] if instr[0] == 'L' else state[0], # Update L if H=='L'
                instr[1] if instr[0] == 'R' else state[1], # Update R if H=='R'
                state[2] + (
                    get_dist(state[0], instr[1], state[1], N) if instr[0] == 'L'
                    else get_dist(state[1], instr[1], state[0], N)
                )
            )
        ),
        instructions,
        initial_state
    )

    print(final_state[2])

if __name__ == "__main__":
    solve()