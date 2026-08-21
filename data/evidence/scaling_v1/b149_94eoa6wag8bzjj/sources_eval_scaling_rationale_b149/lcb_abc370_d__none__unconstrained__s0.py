import sys
from functools import reduce
from bisect import bisect_left

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
    # 1. rows: A list of sorted lists containing indices of existing walls for each row
    # 2. cols: A list of sorted lists containing indices of existing walls for each column
    # 3. total_walls: Current count of walls
    
    # Initial state: every cell has a wall
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    initial_total = H * W
    
    def process_query(state, query):
        rows, cols, total = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # Using bisect to check existence in the sorted list of the row
        idx_in_row = bisect_left(rows[r-1], c)
        has_wall = idx_in_row < len(rows[r-1]) and rows[r-1][idx_in_row] == c
        
        if has_wall:
            # Destroy wall at (r, c)
            # We create new lists to avoid mutation, though pop/remove is faster
            # To keep it loop-free and "functional", we use slicing
            new_rows = list(rows)
            new_rows[r-1] = rows[r-1][:idx_in_row] + rows[r-1][idx_in_row+1:]
            
            new_cols = list(cols)
            idx_in_col = bisect_left(cols[c-1], r)
            new_cols[c-1] = cols[c-1][:idx_in_col] + cols[c-1][idx_in_col+1:]
            
            return (new_rows, new_cols, total - 1)
        else:
            # Destroy first walls in 4 directions
            # Find targets
            row_list = rows[r-1]
            col_list = cols[c-1]
            
            # Left: largest j < c
            idx_l = bisect_left(row_list, c) - 1
            target_l = row_list[idx_l] if idx_l >= 0 else None
            
            # Right: smallest j > c
            idx_r = bisect_left(row_list, c)
            target_r = row_list[idx_r] if idx_r < len(row_list) else None
            
            # Up: largest i < r
            idx_u = bisect_left(col_list, r) - 1
            target_u = col_list[idx_u] if idx_u >= 0 else None
            
            # Down: smallest i > r
            idx_d = bisect_left(col_list, r)
            target_d = col_list[idx_d] if idx_d < len(col_list) else None
            
            targets = [t for t in [target_l, target_r, target_u, target_d] if t is not None]
            
            # To remove multiple walls without loops, we use a helper to update the state
            # Since we can't use loops, we use reduce to apply the "destroy" logic to each target
            def destroy_wall(s, target_info):
                curr_rows, curr_cols, curr_total = s
                tr, tc = target_info
                
                # Remove from row
                r_idx = bisect_left(curr_rows[tr-1], tc)
                new_r = list(curr_rows)
                new_r[tr-1] = curr_rows[tr-1][:r_idx] + curr_rows[tr-1][r_idx+1:]
                
                # Remove from col
                c_idx = bisect_left(curr_cols[tc-1], tr)
                new_c = list(curr_cols)
                new_c[tc-1] = curr_cols[tc-1][:c_idx] + curr_cols[tc-1][c_idx+1:]
                
                return (new_r, new_c, curr_total - 1)

            # Map targets to coordinates
            coords = [
                (r, target_l) if target_l else None,
                (r, target_r) if target_r else None,
                (target_u, c) if target_u else None,
                (target_d, c) if target_d else None
            ]
            valid_coords = [coord for coord in coords if coord]
            
            return reduce(destroy_wall, valid_coords, (rows, cols, total))

    final_state = reduce(process_query, queries, (initial_rows, initial_cols, initial_total))
    print(final_state[2])

if __name__ == "__main__":
    solve()