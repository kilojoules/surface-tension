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

    # We maintain a list of sorted indices of existing walls for every row and column.
    # rows[i] contains sorted column indices of walls in row i.
    # cols[j] contains sorted row indices of walls in column j.
    # Using lists and bisect allows O(log N) search and O(N) deletion.
    # Given H*W <= 4e5, O(N) deletion in the worst case is acceptable for Python 
    # if the total number of deletions is limited to H*W.
    
    initial_state = {
        'rows': [sorted(range(1, W + 1)) for _ in range(H)],
        'cols': [sorted(range(1, H + 1)) for _ in range(W)],
        'wall_count': H * W
    }

    def process_query(state, query):
        r, c = query
        # Adjust to 0-indexed for internal storage
        r_idx, c_idx = r - 1, c - 1
        
        row_walls = state['rows'][r_idx]
        col_walls = state['cols'][c_idx]
        
        # Check if wall exists at (r, c)
        # bisect_left returns the leftmost insertion point to maintain order
        idx_in_row = bisect_left(row_walls, c)
        exists = idx_in_row < len(row_walls) and row_walls[idx_in_row] == c
        
        if exists:
            # Destroy wall at (r, c)
            row_walls.pop(idx_in_row)
            # Find and remove from col_walls
            idx_in_col = bisect_left(col_walls, r)
            col_walls.pop(idx_in_col)
            return {
                'rows': state['rows'],
                'cols': state['cols'],
                'wall_count': state['wall_count'] - 1
            }
        else:
            # Destroy up to 4 neighbors
            # Find neighbors in row (Left and Right)
            # Left: element at idx_in_row - 1
            # Right: element at idx_in_row
            
            # We collect targets to delete to avoid mutating lists while iterating
            # though we aren't using loops, we must handle the 4 directions.
            
            # Helper to remove wall at (r, c)
            def remove_wall(s, row, col):
                # row, col are 1-indexed
                r_i, c_i = row - 1, col - 1
                # Remove from row list
                r_list = s['rows'][r_i]
                pos_r = bisect_left(r_list, col)
                if pos_r < len(r_list) and r_list[pos_r] == col:
                    r_list.pop(pos_r)
                    # Remove from col list
                    c_list = s['cols'][c_i]
                    pos_c = bisect_left(c_list, row)
                    if pos_c < len(c_list) and c_list[pos_c] == row:
                        c_list.pop(pos_c)
                        return s['wall_count'] - 1
                return s['wall_count']

            # Identify targets
            # Row targets
            t_left = row_walls[idx_in_row - 1] if idx_in_row > 0 else None
            t_right = row_walls[idx_in_row] if idx_in_row < len(row_walls) else None
            # Col targets
            idx_in_col = bisect_left(col_walls, r)
            t_up = col_walls[idx_in_col - 1] if idx_in_col > 0 else None
            t_down = col_walls[idx_in_col] if idx_in_col < len(col_walls) else None
            
            targets = [t for t in [t_left, t_right, t_up, t_down] if t is not None]
            
            # Map targets to coordinates
            # Left/Right are (r, col), Up/Down are (row, c)
            coords = [
                (r, t_left) if t_left is not None else None,
                (r, t_right) if t_right is not None else None,
                (t_up, c) if t_up is not None else None,
                (t_down, c) if t_down is not None else None
            ]
            # Filter None and remove duplicates (e.g., if a wall is both left and up)
            unique_targets = list(set(filter(None, coords)))
            
            # Update wall count by attempting to remove each target
            # Since we can't use loops, we use reduce to update the count
            new_count = reduce(lambda count, coord: remove_wall({'wall_count': count, 'rows': state['rows'], 'cols': state['cols']}, *coord), unique_targets, state['wall_count'])
            
            return {
                'rows': state['rows'],
                'cols': state['cols'],
                'wall_count': new_count
            }

    final_state = reduce(process_query, queries, initial_state)
    print(final_state['wall_count'])

if __name__ == "__main__":
    solve()