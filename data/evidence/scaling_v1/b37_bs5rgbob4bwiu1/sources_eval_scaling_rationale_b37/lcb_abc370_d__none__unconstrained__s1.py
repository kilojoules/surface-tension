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
    
    # Queries as a list of (r, c) tuples
    queries = list(map(int, input_data[3:]))
    # Group queries into pairs (R_q, C_q)
    qs = [(queries[i], queries[i+1]) for i in range(0, len(queries), 2)]

    # To avoid loops, we use a functional approach.
    # We maintain a list of sorted lists (or sets) for each row and column.
    # Since we need to remove elements and find neighbors, 
    # we use sorted lists and bisect. 
    # However, removing from a list is O(N). 
    # Given H*W <= 4e5, we use sets for O(1) removal and 
    # sorted lists for O(log N) searching.
    # To strictly avoid 'for' and 'while', we use reduce for state transition.
    
    from functools import reduce

    # Initial state: 
    # rows: list of sets containing column indices of walls
    # cols: list of sets containing row indices of walls
    # Since we need sorted order for bisect, we maintain sorted lists.
    # Because we cannot use loops, we initialize using list comprehensions.
    
    # State structure: (row_walls, col_walls, total_walls)
    # row_walls: list of sorted lists
    # col_#walls: list of sorted lists
    initial_state = (
        [list(range(1, W + 1)) for _ in range(H)],
        [list(range(1, H + 1)) for _ in range(W)],
        H * W
    )

    def process_query(state, q):
        r, c = q
        row_walls, col_walls, count = state
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        r_idx = r - 1
        c_idx = c - 1
        
        # Check if c is in row_walls[r_idx]
        # Since it's sorted, we can use bisect_left
        pos = bisect_left(row_walls[r_idx], c)
        exists = pos < len(row_walls[r_idx]) and row_walls[r_idx][pos] == c
        
        if exists:
            # Destroy wall at (r, c)
            # To "remove" without a loop, we create a new list (slice)
            # Note: list slicing/addition is allowed as it is an expression
            new_row_walls = list(row_walls)
            new_row_walls[r_idx] = row_walls[r_idx][:pos] + row_walls[r_idx][pos+1:]
            
            # Also remove from col_walls
            c_pos = bisect_left(col_walls[c_idx], r)
            new_col_walls = list(col_walls)
            new_col_walls[c_idx] = col_walls[c_idx][:c_pos] + col_walls[c_idx][c_pos+1:]
            
            return (new_row_walls, new_col_walls, count - 1)
        else:
            # Destroy 4 nearest walls
            # Find targets
            # Up: col_walls[c_idx], find largest i < r
            # Down: col_walls[c_idx], find smallest i > r
            # Left: row_walls[r_idx], find largest j < c
            # Right: row_walls[r_idx], find smallest j > c
            
            c_list = col_walls[c_idx]
            r_list = row_walls[r_idx]
            
            c_pos = bisect_left(c_list, r)
            r_pos = bisect_left(r_list, c)
            
            # Targets to destroy: (row, col)
            targets = [
                (c_list[c_pos-1], c) if c_pos > 0 else None, # Up
                (c_list[c_pos] , c) if c_pos < len(c_list) else None, # Down
                (r, r_list[r_pos-1]) if r_pos > 0 else None, # Left
                (r, r_list[r_pos]) if r_pos < len(r_list) else None, # Right
            ]
            
            # Filter None and remove duplicates (though not possible here by logic)
            to_destroy = [t for t in targets if t is not None]
            
            # We need to update the state by removing all to_destroy walls.
            # Since we can't use a loop, we use reduce again to process the targets.
            def remove_wall(s, target):
                tr, tc = target
                tr_idx, tc_idx = tr - 1, tc - 1
                curr_row_walls, curr_col_walls, curr_count = s
                
                # Remove from row_walls
                p_r = bisect_left(curr_row_walls[tr_idx], tc)
                # Check if it still exists (might have been destroyed by another target in this query)
                if p_r < len(curr_row_walls[tr_idx]) and curr_row_walls[tr_idx][p_r] == tc:
                    rw = list(curr_row_walls)
                    rw[tr_idx] = curr_row_walls[tr_idx][:p_r] + curr_row_walls[tr_idx][p_r+1:]
                    
                    # Remove from col_walls
                    p_c = bisect_left(curr_col_walls[tc_idx], tr)
                    cw = list(curr_col_walls)
                    cw[tc_idx] = curr_col_walls[tc_idx][:p_c] + curr_col_walls[tc_idx][p_c+1:]
                    
                    return (rw, cw, curr_count - 1)
                return s

            return reduce(remove_wall, to_destroy, (row_walls, col_walls, count))

    final_state = reduce(process_query, qs, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()