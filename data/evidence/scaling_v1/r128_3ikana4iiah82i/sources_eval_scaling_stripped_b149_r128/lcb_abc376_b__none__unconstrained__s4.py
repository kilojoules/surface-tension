import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H_i, T_i)
    # We use a list comprehension to group the remaining input into pairs
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Initial state: left hand at 1, right hand at 2
    # We use a list to keep track of current positions [left, right]
    current_pos = [1, 2]
    
    # Helper to calculate the shortest distance between two parts on a ring
    # The distance is the minimum of the clockwise and counter-clockwise paths
    def get_dist(start, end):
        return min(abs(start - end), N - abs(start - end))

    # We need to simulate the movement for each instruction.
    # The problem guarantees that the other hand does not block the path.
    # However, the "cannot move through the other hand" rule means we must
    # check if the shortest path is blocked.
    # If the shortest path is blocked, we must take the longer path.
    
    # Let's refine the distance logic:
    # To move from S to E without passing through P:
    # 1. Clockwise distance: (E - S) % N
    # 2. Counter-clockwise distance: (S - E) % N
    # We check if P is "between" S and E for each direction.
    
    def get_min_ops(S, E, P):
        if S == E:
            return 0
        
        # Clockwise distance
        dist_cw = (E - S) % N
        # P is in the clockwise path if (P - S) % N < dist_cw
        blocked_cw = (P - S) % N < dist_cw
        
        # Counter-clockwise distance
        dist_ccw = (S - E) % N
        # P is in the counter-clockwise path if (S - P) % N < dist_ccw
        blocked_ccw = (S - P) % N < dist_ccw
        
        # We are guaranteed the instruction is achievable.
        # If one path is blocked, we must take the other.
        # If neither is blocked, we take the minimum.
        
        res = []
        if not blocked_cw: res.append(dist_cw)
        if not blocked_ccw: res.append(dist_ccw)
        
        return min(res)

    # Process instructions using a loop to update current_pos and accumulate distance
    # Since we cannot use 'for' loops for the logic, we use a functional approach.
    # We can use a custom reduce-like mechanism or a recursive-style mapping.
    # But the constraint says "no for/while loops". 
    # We can use a generator with a mutable state object or a helper function.
    
    state = {"pos": [1, 2], "total": 0}
    
    def process(instr):
        h, t = instr
        idx = 0 if h == 'L' else 1
        other_idx = 1 - idx
        
        s = state["pos"][idx]
        p = state["pos"][other_idx]
        
        ops = get_min_ops(s, t, p)
        state["total"] += ops
        state["pos"][idx] = t
        return None

    # Use map to trigger the process function for all instructions
    list(map(process, instructions))
    
    print(state["total"])

if __name__ == "__main__":
    solve()