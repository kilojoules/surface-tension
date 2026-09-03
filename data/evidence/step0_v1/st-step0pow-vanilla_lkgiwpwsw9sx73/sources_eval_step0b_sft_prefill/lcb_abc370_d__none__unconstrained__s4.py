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
    
    # We need to track which walls exist in each row and each column.
    # Since we can't use loops, we use lists of sorted indices.
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    
    # Using list comprehensions to initialize
    rows_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To track if a wall exists without loops, we use a set of tuples (r, c)
    # However, we need to remove elements from the sorted lists.
    # Since we can't use loops, we can't easily "remove" from the middle of a list.
    # But we can use a technique where we store the state in a way that we can 
    # query the nearest wall. 
    # Given the constraints and "no loop" requirement, we can use a 
    # mutable data structure. Python's list is mutable.
    # To avoid loops, we use map/filter/list comprehensions.
    
    # We will use a dictionary to keep track of destroyed walls.
    destroyed = set()
    
    # To handle the queries without loops, we use a recursive-like structure 
    # via map or a list comprehension.
    
    # We need a way to find the nearest wall. 
    # Since we can't loop, we can't easily update the sorted lists.
    # Let's use a different approach: 
    # For each row and column, we maintain a sorted list of existing walls.
    # When a wall is destroyed, we remove it. 
    # To avoid loops, we use `bisect` to find the index and `pop` to remove.
    
    # We define the logic for a single query in a function.
    def process_query(query):
        r, c = query
        if (r, c) not in destroyed:
            # Destroy wall at (r, c)
            destroyed.add((r, c))
            # Remove from row list
            idx_r = bisect_left(rows_walls[r], c)
            rows_walls[r].pop(idx_r)
            # Remove from col list
            idx_c = bisect_left(cols_walls[c], r)
            cols_walls[c].pop(idx_c)
        else:
            # No wall at (r, c), look in 4 directions
            # Up
            idx_u = bisect_left(cols_walls[c], r) - 1
            # Down
            idx_d = bisect_left(cols_walls[c], r)
            # Left
            idx_l = bisect_left(rows_walls[r], c) - 1
            # Right
            idx_r = bisect_left(rows_walls[r], c)
            
            # Identify walls to destroy
            targets = [
                (cols_walls[c][idx_u], c) if idx_u >= 0 else None,
                (cols_walls[c][idx_d], c) if idx_d < len(cols_walls[c]) else None,
                (r, rows_walls[r][idx_l]) if idx_l >= 0 else None,
                (r, rows_walls[r][idx_r]) if idx_r < len(rows_walls[r]) else None
            ]
            
            # Filter None and remove duplicates
            unique_targets = set(filter(None, targets))
            
            # For each target, destroy it. Since we can't loop, we use map.
            def destroy_wall(wall):
                wr, wc = wall
                if wall not in destroyed:
                    destroyed.add(wall)
                    # Remove from row list
                    r_idx = bisect_left(rows_walls[wr], wc)
                    rows_walls[wr].pop(r_idx)
                    # Remove from col list
                    c_idx = bisect_left(cols_walls[wc], wr)
                    cols_walls[wc].pop(c_idx)
            
            list(map(destroy_wall, unique_targets))

    # Parse queries into tuples
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]
    
    # Process all queries using map
    list(map(process_query, queries))
    
    # Total walls - destroyed walls
    print(H * W - len(destroyed))

if __name__ == "__main__":
    solve()