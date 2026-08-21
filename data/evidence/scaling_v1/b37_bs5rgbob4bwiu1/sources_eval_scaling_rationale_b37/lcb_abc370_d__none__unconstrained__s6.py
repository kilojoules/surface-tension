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

    # We need to track existing walls for each row and each column.
    # rows[i] contains sorted indices of columns that have a wall in row i.
    # cols[j] contains sorted indices of rows that have a wall in column j.
    # Using lists and bisect for efficiency.
    
    # Initial state: All cells have walls.
    # To avoid explicit loops for initialization, we use list comprehensions.
    initial_rows = [sorted(range(1, W + 1)) for _ in range(H)]
    initial_cols = [sorted(range(1, H + 1)) for _ in range(W)]
    
    # State is (rows, cols, total_walls)
    # We use a helper function to handle the logic of a single query.
    def handle_query(state, query):
        rows, cols, total = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # r-1 and c-1 because input is 1-indexed
        row_list = rows[r-1]
        idx = bisect_left(row_list, c)
        has_wall = idx < len(row_list) and row_list[idx] == c
        
        if has_wall:
            # Destroy wall at (r, c)
            # We use slice assignment to simulate removal without a loop
            new_rows = list(rows)
            new_rows[r-1] = row_list[:idx] + row_list[idx+1:]
            
            new_cols = list(cols)
            col_list = cols[c-1]
            c_idx = bisect_left(col_list, r)
            new_cols[c-1] = col_list[:c_idx] + col_list[c_idx+1:]
            
            return (new_rows, new_cols, total - 1)
        else:
            # Destroy up to 4 neighbors
            # Identify targets
            row_list = rows[r-1]
            col_list = cols[c-1]
            
            # Left: largest j < c in row_list
            # Right: smallest j > c in row_list
            # Up: largest i < r in col_list
            # Down: smallest i > r in col_list
            
            idx_r = bisect_left(row_list, c)
            idx_c = bisect_left(col_list, r)
            
            targets = [
                (r, row_list[idx_r-1]) if idx_r > 0 else None, # Left
                (r, row_list[idx_r]) if idx_r < len(row_list) else None, # Right
                (col_list[idx_c-1], c) if idx_c > 0 else None, # Up
                (col_list[idx_c], c) if idx_c < len(col_list) else None, # Down
            ]
            
            # Filter None and remove duplicates (e.g., if H or W is 1)
            valid_targets = list(set(filter(None, targets)))
            
            # To update the state without loops, we use reduce to process the targets
            def remove_wall(s, target):
                curr_rows, curr_cols, curr_total = s
                tr, tc = target
                
                # Remove from row tracking
                r_list = curr_rows[tr-1]
                r_idx = bisect_left(r_list, tc)
                new_r_list = r_list[:r_idx] + r_list[r_idx+1:]
                
                # Remove from col tracking
                c_list = curr_cols[tc-1]
                c_idx = bisect_left(c_list, tr)
                new_c_list = c_list[:c_idx] + c_list[c_idx+1:]
                
                # Update structures
                updated_rows = list(curr_rows)
                updated_rows[tr-1] = new_r_list
                updated_cols = list(curr_cols)
                updated_cols[tc-1] = new_c_list
                
                return (updated_rows, updated_cols, curr_total - 1)

            return reduce(remove_wall, valid_targets, (rows, cols, total))

    # Process all queries using reduce
    final_state = reduce(handle_query, queries, (initial_rows, initial_cols, H * W))
    print(final_state[2])

if __name__ == "__main__":
    solve()