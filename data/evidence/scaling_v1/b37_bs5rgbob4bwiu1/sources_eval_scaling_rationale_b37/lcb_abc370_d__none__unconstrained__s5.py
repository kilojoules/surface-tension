import sys
from bisect import bisect_left, insort

# Increase recursion depth for deep reduce calls if necessary, 
# though reduce is iterative.
sys.setrecursionlimit(200000)

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries as a list of (r, c) tuples
    queries = list(map(int, input_data[3:]))
    query_pairs = list(zip(queries[0::2], queries[1::2]))

    # We maintain a set of existing walls for each row and each column.
    # rows[i] contains sorted column indices of walls in row i.
    # cols[j] contains sorted row indices of walls in column j.
    # Using lists and bisect to simulate a SortedSet.
    
    # Initial state: all cells have walls.
    # To avoid explicit loops, we use list comprehensions.
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # State object to carry through reduce: (rows, cols, total_walls)
    initial_state = (rows, cols, H * W)

    def process_query(state, query):
        rows, cols, total = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # bisect_left finds the index where c would be inserted
        row_list = rows[r-1]
        idx = bisect_left(row_list, c)
        has_wall = idx < len(row_list) and row_list[idx] == c
        
        if has_wall:
            # Destroy wall at (r, c)
            # Remove c from rows[r-1] and r from cols[c-1]
            # We use slice assignment to "remove" without a loop
            # Since we can't use 'del' or 'pop' in a way that avoids 
            # state mutation inside reduce (which is allowed), 
            # but we must avoid 'for/while'.
            
            # Mutation is necessary for performance given H*W <= 4e5
            row_list.pop(idx)
            col_list = cols[c-1]
            c_idx = bisect_left(col_list, r)
            col_list.pop(c_idx)
            return (rows, cols, total - 1)
        else:
            # Destroy up to 4 neighbors
            # 1. Up (same column, smaller row)
            # 2. Down (same column, larger row)
            # 3. Left (same row, smaller column)
            # 4. Right (same row, larger column)
            
            col_list = cols[c-1]
            c_idx = bisect_left(col_list, r)
            
            # Identify targets
            # Up: element at c_idx - 1
            # Down: element at c_idx
            # Left: element in row_list before c
            # Right: element in row_list after c
            
            # We use a helper to handle the removal logic to keep it clean
            def remove_wall(s, r_coord, c_coord):
                # Only remove if wall actually exists
                # We check existence again because simultaneous destruction 
                # might target the same wall.
                r_idx = r_coord - 1
                c_idx_coord = c_coord - 1
                
                r_l = s[0][r_idx]
                c_l = s[1][c_idx_coord]
                
                # Check if wall exists
                pos_in_row = bisect_left(r_l, c_coord)
                if pos_in_row < len(r_l) and r_l[pos_in_row] == c_coord:
                    r_l.pop(pos_in_row)
                    pos_in_col = bisect_left(c_l, r_coord)
                    c_l.pop(pos_in_col)
                    return 1
                return 0

            # Find potential targets
            targets = [
                (col_list[c_idx-1], c) if c_idx > 0 else None, # Up
                (col_list[c_idx], c) if c_idx < len(col_list) else None, # Down
                (r, row_list[idx-1]) if idx > 0 else None, # Left
                (r, row_list[idx]) if idx < len(row_list) else None # Right
            ]
            
            # Filter None and remove walls
            # Since we can't use a loop, we use map/sum and a helper
            # To avoid loops, we process the 4 targets using a list comprehension 
            # and sum the results, though the removal happens via side effect.
            
            # We need to be careful: the targets are identified SIMULTANEOUSLY.
            # So we identify all 4 first, then remove them.
            valid_targets = [t for t in targets if t is not None]
            
            # Use map to call remove_wall for each target
            # We use a list comprehension to trigger the side effect
            removed_count = sum([remove_wall((rows, cols), tr, tc) for tr, tc in valid_targets])
            
            return (rows, cols, total - removed_count)

    from functools import reduce
    final_state = reduce(process_query, query_pairs, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()