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
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # State: rows[r] is a sorted list of columns containing walls in row r
    #        cols[c] is a sorted list of rows containing walls in column c
    # Using 1-based indexing for convenience
    rows = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols = [list(range(1, H + 1)) for _ in range(W + 1)]

    def remove_wall(r, c, state):
        r_list, c_list = state
        # Find index of the wall in the sorted lists
        idx_r = bisect_left(r_list[r], c)
        if idx_r < len(r_list[r]) and r_list[r][idx_r] == c:
            r_list[r].pop(idx_r)
        
        idx_c = bisect_left(c_list[c], r)
        if idx_c < len(c_list[c]) and c_list[c][idx_c] == r:
            c_list[c].pop(idx_c)
        return state

    def process_query(state, query):
        r, c = query
        r_list, c_list = state
        
        # Check if wall exists at (r, c)
        idx_r = bisect_left(r_list[r], c)
        has_wall = idx_r < len(r_list[r]) and r_list[r][idx_r] == c
        
        if has_wall:
            return remove_wall(r, c, state)
        
        # If no wall, find neighbors to destroy
        # Left
        l_idx = idx_r - 1
        left_wall = (r, r_list[r][l_idx]) if l_idx >= 0 else None
        # Right
        r_idx = idx_r
        right_wall = (c, r_list[r][r_idx]) if r_idx < len(r_list[r]) else None
        # Note: the logic above for left/right is slightly wrong because 
        # r_list[r] contains columns. Let's fix:
        
        # Correcting neighbor search:
        # Row r, find nearest columns to c
        # Left:
        l_col = r_list[r][l_idx] if l_idx >= 0 else None
        # Right:
        r_col = r_list[r][r_idx] if r_idx < len(r_list[r]) else None
        
        # Column c, find nearest rows to r
        idx_c = bisect_left(c_list[c], r)
        u_row = c_list[c][idx_c - 1] if idx_c > 0 else None
        d_row = c_list[c][idx_c] if idx_c < len(c_list[c]) else None
        
        # Collect all walls to be destroyed
        targets = [
            (r, l_col), (r, r_col), 
            (u_row, c), (d_row, c)
        ]
        
        # Filter None and apply removal
        # We use reduce to apply remove_wall to all valid targets
        return reduce(
            lambda s, t: remove_wall(t[0], t[1], s) if t[0] and t[1] else s,
            targets,
            state
        )

    # Process all queries
    final_state = reduce(process_query, queries, (rows, cols))
    
    # Calculate remaining walls
    # Sum of lengths of all lists in rows (excluding index 0)
    total_remaining = sum(map(len, final_state[0][1:]))
    print(total_remaining)

if __name__ == "__main__":
    solve()