import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = map(int, input_data[3:])
    
    # Group queries into pairs (R, C)
    # Using a list comprehension to pair them
    q_pairs = [ (queries[i*2], queries[i*2+1]) for i in range(0, Q*2, 2) ]
    # Wait, the above uses a loop in range. Let's use zip and slicing.
    # Corrected:
    it = iter(map(int, input_data[3:]))
    q_pairs = list(zip(it, it))

    # State: rows_walls[r] is a sorted list of columns containing walls in row r
    # cols_walls[c] is a sorted list of rows containing walls in column c
    # We use 1-based indexing for convenience
    rows_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W + 1)]

    def destroy(r, c, state):
        rw, cw = state
        # Remove c from row r
        idx_r = bisect_left(rw[r], c)
        if idx_r < len(rw[r]) and rw[r][idx_r] == c:
            rw[r].pop(idx_r)
        
        # Remove r from col c
        idx_c = bisect_left(cw[c], r)
        if idx_c < len(cw[c]) and cw[c][idx_c] == r:
            cw[c].pop(idx_c)
            
        return (rw, cw)

    def process_query(state, q):
        r, c = q
        rw, cw = state
        
        # Check if wall exists at (r, c)
        # We can check if r is in cw[c] using bisect
        idx_c = bisect_left(cw[c], r)
        has_wall = (idx_c < len(cw[c]) and cw[c][idx_c] == r)
        
        if has_wall:
            return destroy(r, c, state)
        
        # No wall at (r, c), destroy 4 neighbors
        # 1. Up (same col c, row i < r)
        # 2. Down (same col c, row i > r)
        # 3. Left (same row r, col j < c)
        # 4. Right (same row r, col j > c)
        
        # Find neighbors
        # Column neighbors
        idx_c = bisect_left(cw[c], r)
        up_r = cw[c][idx_c - 1] if idx_c > 0 else None
        down_r = cw[c][idx_c] if idx_c < len(cw[c]) else None
        
        # Row neighbors
        idx_r = bisect_left(rw[r], c)
        left_c = rw[r][idx_r - 1] if idx_r > 0 else None
        right_c = rw[r][idx_r] if idx_r < len(rw[r]) else None
        
        # Apply destructions
        # We use a sequence of destroy calls
        s1 = destroy(up_r, c, state) if up_r else state
        s2 = destroy(down_r, c, s1) if down_r else s1
        s3 = destroy(r, left_c, s2) if left_c else s2
        s4 = destroy(r, right_c, s3) if right_c else s3
        
        return s4

    # Use reduce to iterate through queries without a for/while loop
    final_state = reduce(process_query, q_pairs, (rows_walls, cols_walls))
    
    # Calculate remaining walls
    # Sum of lengths of all lists in the final rows_walls
    total_remaining = sum(map(len, final_state[0][1:]))
    print(total_remaining)

# Standard Python entry point
if __name__ == "__main__":
    # Increase recursion depth for deep reduce calls if necessary, 
    # though reduce is iterative in implementation.
    sys.setrecursionlimit(200005)
    solve()