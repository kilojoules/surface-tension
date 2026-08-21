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

    # We maintain sets of existing wall indices for each row and each column.
    # rows[i] contains sorted indices of columns that have a wall in row i.
    # cols[j] contains sorted indices of rows that have a wall in column j.
    # Using lists and bisect for O(log N) search and O(N) deletion.
    # Since H*W is 4e5, we use a dictionary or list of lists.
    
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    def process_query(state, query):
        rows, cols, total_walls = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # r-1 and c-1 because input is 1-indexed
        row_list = rows[r-1]
        idx = bisect_left(row_list, c)
        has_wall = idx < len(row_list) and row_list[idx] == c
        
        if has_wall:
            # Destroy wall at (r, c)
            # We must remove c from rows[r-1] and r from cols[c-1]
            # Using slice assignment to mutate list without a loop
            rows[r-1][idx:idx+1] = []
            col_list = cols[c-1]
            c_idx = bisect_left(col_list, r)
            cols[c-1][c_idx:c_idx+1] = []
            return (rows, cols, total_walls - 1)
        else:
            # Destroy 4 nearest walls
            # 1. Left
            l_idx = idx - 1
            # 2. Right
            r_idx = idx
            # 3. Up
            col_list = cols[c-1]
            u_idx = bisect_left(col_list, r) - 1
            # 4. Down
            d_idx = bisect_left(col_list, r)
            
            # Identify walls to be destroyed
            # We use a list of targets to avoid mutating during the search
            targets = [
                (r, row_list[l_idx]) if l_idx >= 0 else None,
                (r, row_list[r_idx]) if r_idx < len(row_list) else None,
                (col_list[u_idx], c) if u_idx >= 0 else None,
                (col_list[d_idx], c) if d_idx < len(col_list) else None
            ]
            
            # Filter None and remove duplicates (e.g., if a wall is both left and up)
            to_destroy = list(set(filter(None, targets)))
            
            # To remove items from lists without loops, we use a helper 
            # that updates the state for each target.
            def remove_wall(s, target):
                rs, cs, tw = s
                tr, tc = target
                # Remove from row list
                r_l = rs[tr-1]
                r_i = bisect_left(r_l, tc)
                r_l[r_i:r_i+1] = []
                # Remove from col list
                c_l = cs[tc-1]
                c_i = bisect_left(c_l, tr)
                c_l[c_i:c_i+1] = []
                return (rs, cs, tw - 1)
            
            return reduce(remove_wall, to_destroy, (rows, cols, total_walls))

    # Initial state: (rows, cols, total_walls)
    initial_state = (initial_rows, initial_cols, H * W)
    
    # Process all queries using reduce
    final_state = reduce(process_query, queries, initial_state)
    
    # The result is the third element of the final state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()