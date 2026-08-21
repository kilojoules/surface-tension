import sys
from bisect import bisect_left

# Increase recursion depth for deep reduce calls if necessary, 
# though reduce is iterative in implementation.
sys.setrecursionlimit(10**6)

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = [ (int(input_data[i]), int(input_data[i+1])) 
                for i in range(3, len(input_data), 2) ]

    # We maintain sets of active wall indices for each row and column.
    # rows[r] contains sorted columns j where a wall exists at (r, j).
    # cols[c] contains sorted rows i where a wall exists at (i, c).
    # Using lists and bisect for O(log N) search and O(N) deletion.
    # Given H*W <= 4e5, O(N) deletion in a list can be slow, 
    # but since we only delete, it is often acceptable. 
    # However, to be safe and loop-free, we use a state dictionary.
    
    initial_rows = {r: list(range(1, W + 1)) for r in range(1, H + 1)}
    initial_cols = {c: list(range(1, H + 1)) for c in range(1, W + 1)}
    
    def remove_wall(state, r, c):
        # Remove c from row r and r from col c
        # Using list.remove() is O(N), but we must avoid loops.
        # We create new lists to maintain "immutability" logic within reduce.
        # To optimize, we use the fact that we can't use loops, 
        # but we can use slice-based filtering or list comprehensions.
        
        # Update row r
        row_list = state['rows'][r]
        # Use bisect to find index for faster removal
        idx_c = bisect_left(row_list, c)
        new_row = row_list[:idx_c] + row_list[idx_c+1:]
        
        # Update col c
        col_list = state['cols'][c]
        idx_r = bisect_left(col_list, r)
        new_col = col_list[:idx_r] + col_list[idx_r+1:]
        
        # Update the dictionaries
        new_rows = {**state['rows'], r: new_row}
        new_cols = {**state['cols'], c: new_col}
        
        return {'rows': new_rows, 'cols': new_cols, 'count': state['count'] - 1}

    def process_query(state, query):
        r, c = query
        row_list = state['rows'][r]
        
        # Check if wall exists at (r, c)
        # bisect_left returns the index where c would be inserted
        idx = bisect_left(row_list, c)
        exists = idx < len(row_list) and row_list[idx] == c
        
        if exists:
            return remove_wall(state, r, c)
        
        # If no wall, find 4 neighbors
        # Left
        l_idx = idx - 1
        # Right
        r_idx = idx
        # Up (in col c)
        col_list = state['cols'][c]
        u_idx = bisect_left(col_list, r) - 1
        # Down
        d_idx = bisect_left(col_list, r)
        
        # Identify walls to destroy
        targets = [
            (r, row_list[l_idx]) if l_idx >= 0 else None,
            (r, row_list[r_idx]) if r_idx < len(row_list) else None,
            (col_list[u_idx], c) if u_idx >= 0 else None,
            (col_list[d_idx], c) if d_idx < len(col_list) else None
        ]
        
        # Filter None and duplicates, then remove them
        # We use functools.reduce to apply remove_wall to each target
        from functools import reduce
        return reduce(
            lambda s, t: remove_wall(s, t[0], t[1]) if t else s,
            targets,
            state
        )

    from functools import reduce
    
    initial_state = {
        'rows': initial_rows,
        'cols': initial_cols,
        'count': H * W
    }
    
    final_state = reduce(process_query, queries, initial_state)
    print(final_state['count'])

if __name__ == "__main__":
    solve()