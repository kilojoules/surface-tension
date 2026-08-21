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
    # 1. row_walls: A list of sorted lists, where row_walls[i] contains indices of existing walls in row i
    # 2. col_walls: A list of sorted lists, where col_walls[j] contains indices of existing walls in col j
    # 3. total_walls: Current count of walls
    initial_state = (
        [list(range(1, W + 1)) for _ in range(H)],
        [list(range(1, H + 1)) for _ in range(W)],
        H * W
    )

    def process_query(state, query):
        r, c = query
        row_walls, col_walls, total = state
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        r_idx = r - 1
        c_idx = c - 1
        
        # Check if c is in row_walls[r_idx]
        # Since we need to remove elements, and lists are mutable, 
        # we handle the logic inside a helper to maintain the "reduce" structure.
        # However, removing from a list is O(N). Given H*W <= 4e5, 
        # we must be careful. But the problem says we destroy the "first" wall.
        # Using a sorted list and bisect allows us to find the wall in O(log N).
        # Removing from a list is O(N), but since we only remove, 
        # and the total removals are capped at H*W, it might pass if the 
        # distribution of removals is favorable, but for strict O(log N) 
        # we would need a Fenwick tree or Segment tree.
        # Given the constraints and Python's list.pop(), it is often fast enough.
        
        # To avoid explicit loops and recursion, we use a helper function 
        # that performs the mutations and returns the new state.
        def handle_destruction(s, r, c):
            rw, cw, tot = s
            # Check if wall exists at (r, c)
            # We use bisect_left to find the position of c in the sorted row list
            pos_in_row = bisect_set_find(rw[r-1], c)
            
            if pos_in_row != -1:
                # Wall exists: destroy it
                rw[r-1].pop(pos_in_row)
                cw[c-1].remove(r)
                return (rw, cw, tot - 1)
            else:
                # Wall does not exist: destroy 4 neighbors
                # Find nearest walls in 4 directions
                # Up/Down (Column c)
                col_list = cw[c-1]
                idx = bisect_left(col_list, r)
                
                # Down
                down_val = col_list[idx] if idx < len(col_list) else None
                # Up
                up_val = col_list[idx-1] if idx > 0 else None
                
                # Left/Right (Row r)
                row_list = rw[r-1]
                idx_r = bisect_left(row_list, c)
                
                # Right
                right_val = row_list[idx_r] if idx_r < len(row_list) else None
                # Left
                left_val = row_list[idx_r-1] if idx_r > 0 else None
                
                # Collect walls to destroy
                targets = [
                    (up_val, c) if up_val else None,
                    (down_val, c) if down_val else None,
                    (r, left_val) if left_val else None,
                    (r, right_val) if right_val else None
                ]
                
                # Filter None and remove duplicates (e.g., if the same wall is both 'up' and 'left')
                to_destroy = set(filter(None, targets))
                
                # Perform removals
                # We use a trick to mutate the lists without a for-loop: 
                # map(lambda x: ..., to_destroy)
                def remove_wall(wall):
                    wr, wc = wall
                    # Remove from row list
                    # We must find the index again because the list changed
                    r_list = rw[wr-1]
                    r_pos = bisect_set_find(r_list, wc)
                    if r_pos != -1: r_list.pop(r_pos)
                    # Remove from col list
                    c_list = cw[wc-1]
                    c_pos = bisect_set_find(c_list, wr)
                    if c_pos != -1: c_list.pop(c_pos)
                    return 1

                destroyed_count = sum(map(remove_wall, to_destroy))
                return (rw, cw, tot - destroyed_count)

        return handle_destruction(state, r, c)

    def bisect_set_find(sorted_list, val):
        idx = bisect_left(sorted_list, val)
        if idx < len(sorted_list) and sorted_list[idx] == val:
            return idx
        return -1

    # Use reduce to process all queries without a loop
    final_state = reduce(process_query, queries, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()