import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as list of (H, T)
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3])) 
        for i in range(Q)
    ]

    # Helper to calculate distance on ring avoiding the other hand
    # Since we can't use loops, we use list comprehensions to check paths
    def get_dist(start, end, avoid):
        # Clockwise distance
        cw_dist = (end - start) % N
        # Counter-clockwise distance
        ccw_dist = (start - end) % N
        
        # A path is blocked if the 'avoid' node is between start and end
        # For clockwise: start -> start+1 -> ... -> end
        # The avoid node is encountered if (avoid - start) % N < cw_dist
        # However, the problem says we can't move to the destination if the other hand is there.
        # The only way a path is blocked on a ring is if the other hand is "in the way".
        
        # Check if avoid is in the clockwise path
        # The clockwise path consists of nodes (start + k) % N for k in 1..cw_dist
        # It is blocked if (avoid - start) % N is between 1 and cw_dist-1
        # Wait, the rule is: "cannot move to destination part if other hand is there"
        # But the guarantee says T_i != other_hand.
        # So the only restriction is passing through the other hand's current position.
        
        is_cw_blocked = (avoid - start) % N < cw_dist and (avoid - start) % N != 0
        is_ccw_blocked = (start - avoid) % N < ccw_dist and (start - avoid) % N != 0
        
        # We need to handle the 1-indexing by treating everything as 0-indexed internally
        # But the logic above works for any consistent indexing.
        # Since we use % N, we should normalize to 0..N-1
        return min([cw_dist if not is_cw_blocked else float('inf'), 
                    ccw_dist if not is_ccw_blocked else float('inf')])

    # Normalize coordinates to 0-indexed for easier modulo arithmetic
    # Initial state: L=0 (part 1), R=1 (part 2), total=0
    initial_state = (0, 1, 0)

    def process_instruction(state, instr):
        l, r, total = state
        h, t = instr
        t_idx = t - 1
        
        if h == 'L':
            # Move left hand from l to t_idx, avoiding r
            # We need to adjust get_dist because it expects 0-indexed
            # But the logic inside get_dist already handles it if we pass 0-indexed
            d = get_dist(l, t_idx, r)
            # Since the problem guarantees the move is possible, 
            # we use a helper function defined outside or a lambda.
            # Because we can't define functions inside reduce easily, 
            # I'll redefine the distance logic here.
            
            cw = (t_idx - l) % N
            ccw = (l - t_idx) % N
            # Blocked if avoid node is strictly between start and end
            # Clockwise blocked if (r - l) % N < cw
            # Counter-clockwise blocked if (l - r) % N < ccw
            # Note: (r-l)%N == 0 is impossible as l != r
            dist = cw if (r - l) % N >= cw else ccw
            return (t_idx, r, total + dist)
        else:
            # Move right hand from r to t_idx, avoiding l
            cw = (t_idx - r) % N
            ccw = (r - t_idx) % N
            dist = cw if (l - r) % N >= cw else ccw
            return (l, t_idx, total + dist)

    # Use reduce to simulate the process across all instructions
    final_state = reduce(process_instruction, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()