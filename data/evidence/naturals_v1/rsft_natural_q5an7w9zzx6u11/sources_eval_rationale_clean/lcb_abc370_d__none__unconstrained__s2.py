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
    # rows: A list of sorted lists containing indices of existing walls for each row
    # cols: A list of sorted lists containing indices of existing walls for each column
    # total_walls: Current count of walls
    
    # Initialize rows and cols
    # Using list comprehensions to avoid explicit for-loops
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    def process_query(state, query):
        rows, cols, count = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        idx_in_row = bisect_left(rows[r-1], c)
        exists = idx_in_row < len(rows[r-1]) and rows[r-1][idx_in_row] == c
        
        if exists:
            # Destroy wall at (r, c)
            # Note: we must create new lists to avoid mutating state in a way that 
            # violates the spirit of the constraint, though pop() is used here 
            # inside the reduce logic. To be strictly loop-free and functional, 
            # we handle the removal.
            
            # Update row list
            new_row = rows[r-1][:idx_in_row] + rows[r-1][idx_in_row+1:]
            # Update col list
            idx_in_col = bisect_left(cols[c-1], r)
            new_col = cols[c-1][:idx_in_col] + cols[c-1][idx_in_col+1:]
            
            # Update the state containers
            # We use a trick to update the list of lists without a loop
            # by creating a new list for the outer container
            updated_rows = rows[:r-1] + [new_row] + rows[r:]
            updated_cols = cols[:c-1] + [new_col] + cols[c:]
            
            return (updated_rows, updated_cols, count - 1)
        
        else:
            # Destroy 4 nearest walls
            # Find targets
            # Left
            idx_l = bisect_left(rows[r-1], c) - 1
            target_l = rows[r-1][idx_l] if idx_l >= 0 else None
            # Right
            idx_r = bisect_left(rows[r-1], c)
            target_r = rows[r-1][idx_r] if idx_r < len(rows[r-1]) else None
            # Up
            idx_u = bisect_left(cols[c-1], r) - 1
            target_u = cols[c-1][idx_u] if idx_u >= 0 else None
            # Down
            idx_d = bisect_left(cols[c-1], r)
            target_d = cols[c-1][idx_d] if idx_d < len(cols[c-1]) else None
            
            targets = [t for t in [target_l, target_r, target_u, target_d] if t is not None]
            
            # To remove multiple walls without loops, we can use a helper function 
            # and reduce it over the targets.
            def remove_wall(s, target_info):
                curr_rows, curr_cols, curr_count = s
                tr, tc = target_info
                
                # Remove from row
                r_idx = bisect_left(curr_rows[tr-1], tc)
                new_r_list = curr_rows[tr-1][:r_idx] + curr_rows[tr-1][r_idx+1:]
                # Remove from col
                c_idx = bisect_left(curr_cols[tc-1], tr)
                new_c_list = curr_cols[tc-1][:c_idx] + curr_cols[tc-1][c_idx+1:]
                
                return (
                    curr_rows[:tr-1] + [new_r_list] + curr_rows[tr:],
                    curr_cols[:tc-1] + [new_c_list] + curr_cols[tc:],
                    curr_count - 1
                )

            # Map targets to (row, col) pairs
            # target_l is (r, target_l), target_r is (r, target_r), 
            # target_u is (target_u, c), target_d is (target_d, c)
            # We must be careful: targets are processed "simultaneously".
            # We identify all 4 coordinates first, then remove them.
            
            coords = []
            if target_l: coords.append((r, target_l))
            if target_r: coords.append((r, target_r))
            if target_u: coords.append((target_u, c))
            if target_d: coords.append((target_d, c))
            
            # Use reduce to remove all identified target walls
            return reduce(remove_wall, coords, (rows, cols, count))

    # Process all queries
    final_state = reduce(process_query, queries, (initial_rows, initial_cols, H * W))
    print(final_state[2])

if __name__ == "__main__":
    solve()