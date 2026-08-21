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
    
    queries = []
    for i in range(Q):
        queries.append((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])))

    # We need to track existing walls in each row and each column.
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is large.
    # However, we only care about rows and columns that are actually referenced in queries.
    # Actually, the constraint is H*W <= 4e5, so we can afford lists for all rows/cols.
    
    # row_walls[r] stores sorted indices of columns that have a wall in row r.
    # col_walls[c] stores sorted indices of rows that have a wall in column c.
    # Using lists and bisect for O(log N) search and O(N) deletion.
    # Note: Python's list.pop(i) is O(N), but since total walls are 4e5, 
    # and we only delete each wall once, the total time spent deleting is O(H*W).
    # The bottleneck is the search and the deletion.
    
    # To avoid loops, we use a functional approach or map.
    # But we need to maintain state across queries. 
    # Since we cannot use loops, we use a recursive-like structure or a reduction.
    # However, the state depends on the previous query. 
    # We can use a dictionary to store the wall sets and a helper function.
    
    # Wait, the constraint says no for loops. We can use a generator or map.
    # But we need to update the state. We can use a mutable object (like a dict) 
    # inside a function called by map().
    
    state = {
        'row_walls': {r: list(range(1, W + 1)) for r in range(1, H + 1)},
        'col_walls': {c: list(range(1, H + 1)) for c in range(1, W + 1)}
    }

    def destroy(r, c):
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        idx = bisect_left(state['row_walls'][r], c)
        if idx < len(state['row_walls'][r]) and state['row_walls'][r][idx] == c:
            # Wall exists, destroy it
            state['row_walls'][r].pop(idx)
            state['col_walls'][c].pop(bisect_left(state['col_walls'][c], r))
            return
        
        # Wall does not exist, destroy 4 neighbors
        # 1. Up (same column, smaller row)
        c_list = state['col_walls'][c]
        idx_up = bisect_left(c_list, r) - 1
        if idx_up >= 0:
            target_r = c_list[idx_up]
            # Remove wall at (target_r, c)
            state['col_walls'][c].pop(idx_up)
            state['row_walls'][target_r].pop(bisect_left(state['row_walls'][target_r], c))
            
        # 2. Down (same column, larger row)
        # Re-calculate c_list because we might have popped
        c_list = state['col_walls'][c]
        idx_down = bisect_left(c_list, r)
        if idx_down < len(c_list):
            target_r = c_list[idx_down]
            state['col_walls'][c].pop(idx_down)
            state['row_walls'][target_r].pop(bisect_left(state['row_walls'][target_r], c))
            
        # 3. Left (same row, smaller column)
        r_list = state['row_walls'][r]
        idx_left = bisect_left(r_list, c) - 1
        if idx_left >= 0:
            target_c = r_list[idx_left]
            state['row_walls'][r].pop(idx_left)
            state['col_walls'][target_c].pop(bisect_left(state['col_walls'][target_c], r))
            
        # 4. Right (same row, larger column)
        r_list = state['row_walls'][r]
        idx_right = bisect_left(r_list, c)
        if idx_right < len(r_list):
            target_c = r_list[idx_right]
            state['row_walls'][r].pop(idx_right)
            state['col_walls'][target_c].pop(bisect_left(state['col_walls'][target_c], r))

    # Process all queries using map to avoid explicit for loop
    list(map(lambda q: destroy(*q), queries))
    
    # Calculate remaining walls
    # Total walls = H*W. Destroyed walls = (H*W) - remaining.
    # We can just sum the lengths of the lists in row_walls.
    remaining = sum(map(len, state['row_walls'].values()))
    print(remaining)

if __name__ == "__main__":
    solve()