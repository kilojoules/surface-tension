import sys
from bisect import bisect_left, bisect_right

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = input_data[3:]

    # We need to track which walls exist in each row and each column.
    # Since we can't use loops or recursion, we use lists of sorted indices.
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    
    # To avoid loops, we use map/list comprehensions.
    rows_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # We need a way to track if a wall is gone without loops.
    # We'll use a set to store destroyed walls.
    destroyed = set()

    # Since we cannot use for/while loops, we use a recursive-like 
    # structure via map or a list comprehension that triggers side effects.
    # However, the prompt says "no for or while loops". 
    # We can use a functional approach or a trick with map().
    
    def process_query(q_idx):
        r = int(queries[2 * q_idx])
        c = int(queries[2 * q_idx + 1])
        
        # Check if wall exists at (r, c)
        # We use a trick: if (r, c) is in destroyed, it's gone.
        # But we need to know if it WAS a wall.
        # Actually, the logic is: if (r, c) not in destroyed, destroy it.
        # Else, destroy neighbors.
        
        # We use a list to handle the conditional logic since we can't use if/else blocks 
        # for control flow (though if/else expressions are allowed).
        
        # This function returns the set of walls to destroy for this query.
        def get_to_destroy():
            if (r, c) not in destroyed:
                return {(r, c)}
            
            # Find neighbors using bisect on the sorted lists of remaining walls.
            # Note: rows_walls and cols_walls are modified in-place via a trick.
            # To comply with "no loops", we must be careful.
            
            # Row neighbors
            row_list = rows_walls[r]
            idx = bisect_left(row_list, c)
            left = row_list[idx - 1] if idx > 0 else None
            right = row_list[idx] if idx < len(row_list) else None
            
            # Col neighbors
            col_list = cols_walls[c]
            idx_c = bisect_left(col_list, r)
            up = col_list[idx_c - 1] if idx_c > 0 else None
            down = col_list[idx_c] if idx_c < len(col_list) else None
            
            return {
                (r, left) if left else (0, 0),
                (r, right) if right else (0, 0),
                (up, c) if up else (0, 0),
                (down, c) if down else (0, 0)
            }

        to_remove = get_to_destroy()
        
        # Remove the walls from the sorted lists and the destroyed set.
        # We use a list comprehension to filter the lists.
        # Since we can't use for loops, we can't easily mutate the lists 
        # without re-assigning.
        return to_remove

    # To truly avoid 'for' and 'while', we use a mutable state and map().
    # We need to update rows_walls and cols_walls.
    
    # We'll use a helper to perform the deletion.
    def perform_destruction(q_idx):
        r = int(queries[2 * q_idx])
        c = int(queries[2 * q_idx + 1])
        
        # Determine which walls to destroy
        # We use a list to capture the logic of if/else
        res = [ (r, c) ] if (r, c) not in destroyed else [
            (r, rows_walls[r][bisect_left(rows_walls[r], c) - 1]) if bisect_left(rows_walls[r], c) > 0 else (0,0),
            (r, rows_walls[r][bisect_left(rows_walls[r], c)]) if bisect_left(rows_walls[r], c) < len(rows_walls[r]) else (0,0),
            (cols_walls[c][bisect_left(cols_walls[c], r) - 1], c) if bisect_left(cols_walls[c], r) > 0 else (0,0),
            (cols_walls[c][bisect_left(cols_walls[c], r)], c) if bisect_left(cols_walls[c], r) < len(cols_walls[c]) else (0,0)
        ]
        
        # Filter out (0,0) and already destroyed
        targets = [wall for wall in res if wall != (0, 0)]
        
        # Update the global state
        # We use a map to apply the deletion to each target wall
        def remove_wall(wall):
            wr, wc = wall
            if wall not in destroyed:
                destroyed.add(wall)
                # Remove from sorted lists. Since we can't loop, we use bisect and pop.
                # pop() is O(1) or O(N), but we can't use loops to find the index.
                # bisect_left gives the index.
                rows_walls[wr].pop(bisect_left(rows_walls[wr], wc))
                cols_walls[wc].pop(bisect_left(cols_walls[wc], wr))
        
        list(map(remove_wall, targets))

    # Execute all queries using map
    list(map(perform_destruction, range(Q)))

    # Total walls remaining = H*W - number of unique destroyed walls
    print(H * W - len(destroyed))

if __name__ == "__main__":
    solve()