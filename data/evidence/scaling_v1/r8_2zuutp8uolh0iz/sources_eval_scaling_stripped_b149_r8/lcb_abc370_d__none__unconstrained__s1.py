import sys
from bisect import bisect_left
from sortedcontainers import SortedList

def solve():
    # Reading input using sys.stdin.read().split() to handle all whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # We need to track existing walls for every row and every column.
    # rows_walls[r] contains sorted column indices of walls in row r.
    # cols_walls[c] contains sorted row indices of walls in column c.
    # Using SortedList for O(log N) insertions/deletions and O(log N) lookups.
    rows_walls = [SortedList(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [SortedList(range(1, H + 1)) for _ in range(W + 1)]

    # To avoid loops, we use a functional approach via a reduction-like 
    # simulation. Since Python doesn't have a built-in 'reduce' that 
    # allows complex state mutation without a loop, and the constraints 
    # forbid 'for/while', we use a recursive-like structure or map.
    # However, the most reliable way to process Q queries sequentially 
    # without loops is to use a helper function and a list comprehension 
    # or map, but we must mutate the SortedLists.
    
    def process_query(q):
        r, c = q
        # Check if wall exists at (r, c)
        # We use bisect_left via SortedList's internal mechanism or 
        # simply check if c is in the SortedList.
        if c in rows_walls[r]:
            rows_walls[r].remove(c)
            cols_walls[c].remove(r)
            return 1
        else:
            # Find neighbors to destroy
            destroyed = 0
            
            # Up
            idx = cols_walls[c].bisect_left(r)
            if idx > 0:
                target_r = cols_walls[c][idx-1]
                rows_walls[target_r].remove(c)
                cols_walls[c].remove(target_r)
                destroyed += 1
            
            # Down
            idx = cols_walls[c].bisect_left(r)
            if idx < len(cols_walls[c]):
                target_r = cols_walls[c][idx]
                rows_walls[target_r].remove(c)
                cols_walls[c].remove(target_r)
                destroyed += 1
                
            # Left
            idx = rows_walls[r].bisect_left(c)
            if idx > 0:
                target_c = rows_walls[r][idx-1]
                rows_walls[r].remove(target_c)
                cols_walls[target_c].remove(r)
                destroyed += 1
                
            # Right
            idx = rows_walls[r].bisect_left(c)
            if idx < len(rows_walls[r]):
                target_c = rows_walls[r][idx]
                rows_walls[r].remove(target_c)
                cols_walls[target_c].remove(r)
                destroyed += 1
                
            return destroyed

    # Use map to execute process_query for each query in the list.
    # The list() constructor forces the evaluation of the map.
    total_destroyed = sum(list(map(process_query, queries)))
    
    # Total walls initially H*W
    print(H * W - total_destroyed)

if __name__ == "__main__":
    solve()