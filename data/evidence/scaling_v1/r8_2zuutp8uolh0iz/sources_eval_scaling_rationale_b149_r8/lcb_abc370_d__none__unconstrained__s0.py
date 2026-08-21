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

    # State consists of:
    # rows: A list of sorted lists containing column indices of existing walls for each row
    # cols: A list of sorted lists containing row indices of existing walls for each column
    # total_walls: Current count of walls
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    initial_state = (initial_rows, initial_cols, H * W)

    def process_query(state, query):
        rows, cols, total = state
        r, c = query
        
        # Adjust to 0-indexed for internal lists
        r_idx, c_idx = r - 1, c - 1
        
        # Check if wall exists at (r, c)
        # We use bisect_left to check existence in O(log N)
        row_walls = rows[r_idx]
        pos = bisect_left(row_walls, c)
        exists = pos < len(row_walls) and row_walls[pos] == c
        
        if exists:
            # Destroy wall at (r, c)
            # Note: Using slice assignment to "mutate" the list without a loop
            # since the constraint forbids 'for/while' but allows list operations.
            # However, since we need to return a new state for reduce, 
            # we handle the removal carefully.
            new_rows = list(rows)
            new_rows[r_idx] = row_walls[:pos] + row_walls[pos+1:]
            
            new_cols = list(cols)
            col_walls = cols[c_idx]
            c_pos = bisect_left(col_walls, r)
            new_cols[c_idx] = col_walls[:c_pos] + col_walls[c_pos+1:]
            
            return (new_rows, new_cols, total - 1)
        else:
            # Destroy first walls in 4 directions
            # We identify the targets first
            row_walls = rows[r_idx]
            col_walls = cols[c_idx]
            
            # Left: largest index < c
            # Right: smallest index > c
            # Up: largest index < r
            # Down: smallest index > r
            
            # Using bisect to find neighbors
            # Left
            l_pos = bisect_left(row_walls, c) - 1
            target_l = (r, row_walls[l_pos]) if l_pos >= 0 else None
            # Right
            r_pos = bisect_left(row_walls, c)
            target_r = (r, row_walls[r_pos]) if r_pos < len(row_walls) else None
            # Up
            u_pos = bisect_left(col_walls, r) - 1
            target_u = (col_walls[u_pos], c) if u_pos >= 0 else None
            # Down
            d_pos = bisect_left(col_walls, r)
            target_d = (col_walls[d_pos], c) if d_pos < len(col_walls) else None
            
            targets = [t for t in [target_l, target_r, target_u, target_d] if t]
            
            # To avoid loops, we use a nested reduce to remove all targets from the state
            def remove_wall(s, target):
                curr_rows, curr_cols, curr_total = s
                tr, tc = target
                tr_idx, tc_idx = tr - 1, tc - 1
                
                # Remove from row
                r_w = curr_rows[tr_idx]
                p_r = bisect_left(r_w, tc)
                new_r_w = r_w[:p_r] + r_w[p_r+1:]
                
                # Remove from col
                c_w = curr_cols[tc_idx]
                p_c = bisect_left(c_w, tr)
                new_c_w = c_w[:p_c] + c_w[p_c+1:]
                
                # Update state
                res_rows = list(curr_rows)
                res_rows[tr_idx] = new_r_w
                res_cols = list(curr_cols)
                res_cols[tc_idx] = new_c_w
                
                return (res_rows, res_cols, curr_total - 1)

            return reduce(remove_wall, targets, state)

    final_state = reduce(process_query, queries, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    # Increase recursion depth for deep reduce calls if necessary, 
    # though reduce is iterative in Python.
    sys.setrecursionlimit(10**6)
    solve()