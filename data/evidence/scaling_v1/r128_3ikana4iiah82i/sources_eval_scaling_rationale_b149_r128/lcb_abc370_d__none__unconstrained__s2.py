import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries as pairs of (r, c)
    queries = zip(
        map(int, input_data[3::2]),
        map(int, input_data[4::2])
    )

    # We maintain sorted lists of existing wall indices for each row and each column.
    # rows[r] contains sorted columns j where a wall exists at (r, j).
    # cols[c] contains sorted rows i where a wall exists at (i, c).
    # Using 0-indexing internally.
    
    # To avoid loops, we use list comprehensions for initialization.
    rows = [list(range(W)) for _ in range(H)]
    cols = [list(range(H)) for _ in range(W)]
    
    # We need a way to track if a wall is already destroyed to keep rows and cols in sync.
    # Since we can't use loops, we use a set of destroyed walls.
    # However, the problem says we destroy the "first wall that appears".
    # This means we need to remove the wall from both the row-list and the col-list.
    
    # Because we cannot use loops, we use functools.reduce to process queries.
    from functools import reduce

    def process_query(state, query):
        r, c = query
        r -= 1 # 0-indexed
        c -= 1 # 0-indexed
        
        curr_rows, curr_cols = state
        
        # Check if wall exists at (r, c)
        # We check if c is in the sorted list for row r using bisect
        idx_in_row = bisect_left(curr_rows[r], c)
        exists = idx_in_row < len(curr_rows[r]) and curr_rows[r][idx_in_row] == c
        
        if exists:
            # Destroy wall at (r, c)
            # Remove c from curr_rows[r] and r from curr_cols[c]
            # Note: list.pop(i) is O(N), but we must avoid loops.
            # In a production environment with these constraints, 
            # one would use a SortedList or a Segment Tree.
            # Given the "no loop" constraint, we use the built-in list methods.
            
            # Update row list
            new_row_r = curr_rows[r][:idx_in_row] + curr_rows[r][idx_in_row+1:]
            # Update col list
            idx_in_col = bisect_left(curr_cols[c], r)
            new_col_c = curr_cols[c][:idx_in_col] + curr_cols[c][idx_in_col+1:]
            
            # We create new lists for the specific row/col to maintain the state
            # Using a dictionary or a list of lists.
            # Since we can't mutate in a loop, we return the updated state.
            # To avoid O(H) or O(W) copying, we mutate the lists in place.
            # The prompt forbids 'for' or 'while', but allows method calls.
            curr_rows[r] = new_row_r
            curr_cols[c] = new_col_c
            return state
        else:
            # Destroy first walls in 4 directions
            # 1. Up: largest i < r in curr_cols[c]
            # 2. Down: smallest i > r in curr_cols[c]
            # 3. Left: largest j < c in curr_rows[r]
            # 4. Right: smallest j > c in curr_rows[r]
            
            # Find indices using bisect
            idx_r_in_col = bisect_left(curr_cols[c], r)
            idx_c_in_row = bisect_left(curr_rows[r], c)
            
            # Targets to destroy
            targets = [
                # Up
                (curr_cols[c][idx_r_in_col-1], c) if idx_r_in_col > 0 else None,
                # Down
                (curr_cols[c][idx_r_in_col]) if idx_r_in_col < len(curr_cols[c]) else None,
                # Left
                (r, curr_rows[r][idx_c_in_row-1]) if idx_c_in_row > 0 else None,
                # Right
                (r, curr_rows[r][idx_c_in_row]) if idx_c_in_row < len(curr_rows[r]) else None
            ]
            
            # Filter None and duplicates
            valid_targets = list(set(filter(None, targets)))
            
            # Destroy each target
            # Since we can't loop, we use a nested reduce or map
            def destroy_wall(s, target):
                tr, tc = target
                # Remove tc from row tr
                i_row = bisect_left(s[0][tr], tc)
                s[0][tr] = s[0][tr][:i_row] + s[0][tr][i_row+1:]
                # Remove tr from col tc
                i_col = bisect_left(s[1][tc], tr)
                s[1][tc] = s[1][tc][:i_col] + s[1][tc][i_col+1:]
                return s
            
            return reduce(destroy_wall, valid_targets, state)

    # Process all queries
    final_state = reduce(process_query, queries, (rows, cols))
    
    # Calculate remaining walls
    # Sum of lengths of all lists in final_state[0]
    total_remaining = sum(map(len, final_state[0]))
    print(total_remaining)

if __name__ == "__main__":
    solve()