import sys
from bisect import bisect_left, bisect_right
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries as pairs of (r, c)
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # We maintain the state of the walls using:
    # 1. A set of all currently existing walls (r, c)
    # 2. For each row, a sorted list of columns that have walls
    # 3. For each column, a sorted list of rows that have walls
    # However, since we cannot use loops to update lists, we use a dictionary 
    # of sets for rows and columns and handle the "sorted" requirement 
    # by calling sorted() inside the logic.
    
    # To avoid loops, we use reduce to process queries.
    # State: (current_walls_set, row_walls_dict, col_walls_dict)
    # row_walls_dict: {row_idx: set(col_indices)}
    # col_walls_dict: {col_idx: set(row_indices)}
    
    initial_state = (
        set((r, c) for r in range(1, H + 1) for c in range(1, W + 1)),
        {r: set(range(1, W + 1)) for r in range(1, H + 1)},
        {c: set(range(1, H + 1)) for c in range(1, H + 1)}
    )

    def process_query(state, query):
        walls, row_map, col_map = state
        r, c = query
        
        if (r, c) in walls:
            # Destroy wall at (r, c)
            walls.remove((r, c))
            row_map[r].remove(c)
            col_map[c].remove(r)
            return (walls, row_map, col_map)
        
        # No wall at (r, c), destroy neighbors
        # Find neighbors using sorted lists and bisect
        row_walls = sorted(row_map[r])
        col_walls = sorted(col_map[c])
        
        idx_r = bisect_left(row_walls, c)
        idx_c = bisect_left(col_walls, r)
        
        # Potential walls to destroy
        # Left: row_walls[idx_r - 1] if idx_r > 0
        # Right: row_walls[idx_r] if idx_r < len(row_walls)
        # Up: col_walls[idx_c - 1] if idx_c > 0
        # Down: col_walls[idx_c] if idx_c < len(col_walls)
        
        targets = {
            (r, row_walls[idx_r - 1]) if idx_r > 0 else None,
            (r, row_walls[idx_r]) if idx_r < len(row_walls) else None,
            (col_walls[idx_c - 1], c) if idx_c > 0 else None,
            (col_walls[idx_c], c) if idx_c < len(col_walls) else None
        }
        
        # Remove None and update state
        # We use a side-effect inside a list comprehension to update the sets
        # because we cannot use for-loops.
        actual_targets = [t for t in targets if t is not None]
        
        # Update walls set
        walls.difference_update(actual_targets)
        
        # Update row_map and col_map
        # Since we can't loop, we use a trick with map or a comprehension
        # to trigger the .remove() method.
        [row_map[tr].remove(tc) for tr, tc in actual_targets]
        [col_map[tc].remove(tr) for tr, tc in actual_targets]
        
        return (walls, row_map, col_map)

    final_state = reduce(process_query, queries, initial_state)
    print(len(final_state[0]))

if __name__ == "__main__":
    solve()