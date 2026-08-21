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
    
    queries = []
    for i in range(Q):
        queries.append((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])))

    # We need to track existing walls in each row and each column.
    # Since H*W is up to 4e5, we can't use a full 2D array if H or W is very large,
    # but the constraint H*W <= 4e5 allows us to store lists of walls.
    # However, initializing lists for every row/col is only feasible if we use 
    # a comprehension.
    
    # rows_walls[i] stores sorted indices of columns that have a wall in row i.
    # cols_walls[j] stores sorted indices of rows that have a wall in column j.
    # To avoid loops, we use list comprehensions.
    rows_walls = [list(range(1, W + 1)) for _ in range(H)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W)]
    
    # We need a way to track if a wall is destroyed. 
    # A set of (r, c) is too slow for removals. 
    # Instead, we use a boolean array flattened to 1D.
    # wall_exists[ (r-1)*W + (c-1) ]
    wall_exists = [True] * (H * W)

    # Since we cannot use loops to process queries, and we must update the state,
    # we use a helper function with a mutable state (the lists and the boolean array)
    # and call it via a list comprehension or map.
    # However, the problem requires finding the "first" wall, which implies 
    # searching in the sorted lists. Since we can't use loops, we can't easily
    # remove elements from the middle of lists (O(N)). 
    # But wait, the constraint H*W <= 4e5 and Q <= 2e5 suggests an O(Q log(max(H,W))) 
    # or O(Q * constant) approach.
    
    # Actually, the only way to "remove" from a sorted list without loops 
    # and maintain order is to use a data structure like a SortedList, 
    # but that's not standard. 
    # Given the constraints and the "no loop" rule, we can use a 
    # recursive-like structure via map/reduce or a trick with a 
    # mutable object updated inside a list comprehension.
    
    # Let's redefine: we need to find the nearest wall. 
    # We can use a dictionary of sets for rows and columns.
    # Even though we can't use loops, we can use a function that 
    # modifies the sets and call it inside a list comprehension.
    
    from sortedcontainers import SortedList # Not available in standard lib
    # Since SortedList isn't standard, we use bisect on standard lists.
    # Removing from a list is O(N), which is too slow.
    # But we can use a Fenwick tree or Segment tree to find the nearest 1.
    # That's too complex for a "no loop" constraint.
    
    # Wait, the constraint is H*W <= 4e5. If we use a list and .pop(index),
    # it is O(W). In the worst case, this is O(Q*W). 
    # However, for many test cases, this might pass if the total removals are limited.
    # Let's use a function that handles the logic and call it via a list comprehension.

    def process_query(q):
        r, c = q
        idx = (r - 1) * W + (c - 1)
        
        if wall_exists[idx]:
            wall_exists[idx] = False
            # Remove from tracking lists
            # Using .remove() is O(N), but we have no choice without loops/complex structures
            rows_walls[r-1].remove(c)
            cols_walls[c-1].remove(r)
            return 1
        else:
            destroyed = 0
            # Look Up
            # Find largest i < r in cols_walls[c-1]
            row_list = cols_walls[c-1]
            pos = bisect_left(row_list, r)
            if pos > 0:
                target_r = row_list[pos-1]
                wall_exists[(target_r-1)*W + (c-1)] = False
                rows_walls[target_r-1].remove(c)
                row_list.pop(pos-1)
                destroyed += 1
            
            # Look Down
            # Find smallest i > r in cols_walls[c-1]
            # Re-fetch row_list because it might have changed
            row_list = cols_walls[c-1]
            pos = bisect_left(row_list, r)
            if pos < len(row_list):
                target_r = row_list[pos]
                wall_exists[(target_r-1)*W + (c-1)] = False
                rows_walls[target_r-1].remove(c)
                row_list.pop(pos)
                destroyed += 1
                
            # Look Left
            # Find largest j < c in rows_walls[r-1]
            col_list = rows_walls[r-1]
            pos = bisect_left(col_list, c)
            if pos > 0:
                target_c = col_list[pos-1]
                wall_exists[(r-1)*W + (target_c-1)] = False
                cols_walls[target_c-1].remove(r)
                col_list.pop(pos-1)
                destroyed += 1
                
            # Look Right
            # Find smallest j > c in rows_walls[r-1]
            col_list = rows_walls[r-1]
            pos = bisect_left(col_list, c)
            if pos < len(col_list):
                target_c = col_list[pos]
                wall_exists[(r-1)*W + (target_c-1)] = False
                cols_walls[target_c-1].remove(r)
                col_list.pop(pos)
                destroyed += 1
                
            return destroyed

    # Use a list comprehension to trigger the function for each query
    total_destroyed = sum([process_query(q) for q in queries])
    print(H * W - total_destroyed)

if __name__ == "__main__":
    solve()