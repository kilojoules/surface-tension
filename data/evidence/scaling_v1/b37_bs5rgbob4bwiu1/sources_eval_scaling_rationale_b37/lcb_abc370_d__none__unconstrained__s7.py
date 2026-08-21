import sys
from bisect import bisect_left

def solve():
    # Read H, W, Q from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries as a list of (r, c) tuples
    queries = list(map(int, input_data[3:]))
    query_pairs = list(zip(queries[0::2], queries[1::2]))

    # row_walls[i] contains sorted indices of columns that have a wall in row i
    # col_walls[j] contains sorted indices of rows that have a wall in column j
    # Using 1-based indexing for rows and columns to match input
    row_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    col_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To track if a wall exists at (r, c) efficiently
    # Since H*W is up to 4e5, a set of (r, c) is viable, 
    # but we can also check if c is in row_walls[r] using bisect.
    # However, removing from a list is O(N). Given the constraints and the 
    # nature of the problem, we need a way to remove elements.
    # Since we cannot use loops, we use a functional approach to state.
    # Actually, the prompt forbids 'for' and 'while', but allows 'map', 'reduce', etc.
    # To handle removals from sorted lists without loops, we can use slice assignment
    # or list comprehension inside a reduce function.
    
    from functools import reduce

    def process_query(state, query):
        r, c = query
        row_lists, col_lists = state
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        idx = bisect_left(row_lists[r], c)
        exists = idx < len(row_lists[r]) and row_lists[r][idx] == c
        
        if exists:
            # Destroy wall at (r, c)
            # Update row_lists[r] and col_lists[c]
            # Using slice assignment to simulate removal
            new_row_list = row_lists[r][:idx] + row_lists[r][idx+1:]
            
            # Find index in col_lists[c] to remove r
            c_idx = bisect_left(col_lists[c], r)
            new_col_list = col_lists[c][:c_idx] + col_lists[c][c_idx+1:]
            
            # Update state
            row_lists[r] = new_row_list
            col_lists[c] = new_col_list
            return (row_lists, col_lists)
        else:
            # Destroy first walls in 4 directions
            # Up: row < r, same c. Down: row > r, same c.
            # Left: col < c, same r. Right: col > c, same r.
            
            # Column search (Up/Down)
            c_idx = bisect_left(col_lists[c], r)
            # Up
            up_r = col_lists[c][c_idx - 1] if c_idx > 0 else None
            # Down
            down_r = col_lists[c][c_idx] if c_idx < len(col_lists[c]) else None
            
            # Row search (Left/Right)
            r_idx = bisect_left(row_lists[r], c)
            # Left
            left_c = row_lists[r][r_idx - 1] if r_idx > 0 else None
            # Right
            right_c = row_lists[r][r_idx] if r_idx < len(row_lists[r]) else None
            
            # We need to remove up to 4 walls. 
            # Since we must avoid loops, we use a helper to remove a wall.
            def remove_wall(s, row, col):
                rl, cl = s
                if row is None or col is None: return s
                # Remove col from row_lists[row]
                idx_r = bisect_left(rl[row], col)
                if idx_r < len(rl[row]) and rl[row][idx_r] == col:
                    rl[row] = rl[row][:idx_r] + rl[row][idx_r+1:]
                    # Remove row from col_lists[col]
                    idx_c = bisect_left(cl[col], row)
                    if idx_c < len(cl[col]) and cl[col][idx_c] == row:
                        cl[col] = cl[col][:idx_c] + cl[col][idx_c+1:]
                return (rl, cl)

            # Chain the removals using reduce
            targets = [(up_r, c), (down_r, c), (r, left_c), (r, right_c)]
            return reduce(lambda s, t: remove_wall(s, t[0], t[1]), targets, (row_lists, col_lists))

    # Initial state
    # Correcting row_walls/col_walls to be lists of lists
    # We use a list for row_walls and col_walls and mutate them inside the reduce
    # (Though mutation is imperative, the structure is functional)
    initial_state = (
        [list(range(1, W + 1)) for _ in range(H + 1)],
        [list(range(1, H + 1)) for _ in range(W + 1)]
    )
    
    final_state = reduce(process_query, query_pairs, initial_state)
    
    # Calculate remaining walls
    # Sum of lengths of all lists in row_walls
    total_remaining = sum(map(len, final_state[0][1:]))
    print(total_remaining)

if __name__ == "__main__":
    solve()