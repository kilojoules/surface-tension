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

    # We maintain sorted lists of active wall indices for each row and each column.
    # rows[i] contains columns j where a wall exists at (i, j)
    # cols[j] contains rows i where a wall exists at (i, j)
    # Using 0-indexing internally.
    
    # Initial state: all walls present.
    # To avoid explicit loops for initialization, we use list comprehensions.
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # We need a way to track if a wall is destroyed to update both row and col lists.
    # Since we cannot use loops, we use a set of destroyed walls.
    # However, updating sorted lists requires mutation. 
    # We will use a helper function inside reduce.
    
    def process_query(state, query):
        rows, cols, destroyed_count = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        row_list = rows[r-1]
        idx = bisect_left(row_list, c)
        
        if idx < len(row_list) and row_list[idx] == c:
            # Wall exists, destroy it
            row_list.pop(idx)
            # To destroy from cols[c-1], we need the index of r in that list
            col_list = cols[c-1]
            c_idx = bisect_left(col_list, r)
            col_list.pop(c_idx)
            return (rows, cols, destroyed_count + 1)
        else:
            # Wall does not exist, destroy 4 neighbors
            # 1. Up (same col, smaller row)
            # 2. Down (same col, larger row)
            # 3. Left (same row, smaller col)
            # 4. Right (same row, larger col)
            
            # We use a list of targets to destroy to handle them uniformly
            # Target: (row, col)
            
            # Find neighbors
            col_list = cols[c-1]
            c_idx = bisect_left(col_list, r)
            
            # Up: col_list[c_idx-1] if c_idx > 0
            # Down: col_list[c_idx] if c_idx < len(col_list)
            # Left: row_list[idx-1] if idx > 0
            # Right: row_list[idx] if idx < len(row_list)
            
            targets = [
                (col_list[c_idx-1], c) if c_idx > 0 else None,
                (col_list[c_idx], c) if c_idx < len(col_list) else None,
                (r, row_list[idx-1]) if idx > 0 else None,
                (r, row_list[idx]) if idx < len(row_list) else None
            ]
            
            # Filter None and remove duplicates (e.g., if H=1 or W=1)
            valid_targets = list(set(filter(None, targets)))
            
            # We need to remove these walls from both row and col lists.
            # Since we can't loop, we use reduce again to process the targets.
            def remove_wall(s, target):
                tr, tc = target
                r_list = s[0][tr-1]
                c_list = s[1][tc-1]
                
                # Remove from row list
                r_idx = bisect_left(r_list, tc)
                if r_idx < len(r_list) and r_list[r_idx] == tc:
                    r_list.pop(r_idx)
                
                # Remove from col list
                c_idx_rem = bisect_left(c_list, tr)
                if c_idx_rem < len(c_list) and c_list[c_idx_rem] == tr:
                    c_list.pop(c_idx_rem)
                
                return (s[0], s[1], s[2] + 1)

            return reduce(remove_wall, valid_targets, (rows, cols, destroyed_count))

    # Final state calculation
    final_state = reduce(process_query, queries, (initial_rows, initial_cols, 0))
    total_walls = H * W
    print(total_walls - final_state[2])

if __name__ == "__main__":
    solve()