```python
import sys

def solve():
    # Increase recursion depth for deep trees if necessary, 
    # though we use an iterative approach with DSU.
    sys.setrecursionlimit(10**6)
    
    # Read H, W, Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Since H*W can be up to 4*10^5, we can't use a 2D array for DSU 
    # if we want to avoid memory overhead or use specific structures.
    # We use DSU to find the nearest wall in four directions.
    # For each row, we maintain DSU for left/right.
    # For each column, we maintain DSU for up/down.
    
    # wall_exists[r][c] tracks if a wall is present.
    # To save memory, we use a 1D array: index = r * W + c
    wall_exists = [True] * (H * W)
    
    # DSU structures to skip empty cells
    # row_next[r][c] -> next wall to the right
    # row_prev[r][c] -> next wall to the left
    # col_next[r][c] -> next wall down
    # col_prev[r][c] -> next wall up
    
    # Instead of full DSU for every cell, we can use a simple 
    # linked-list style approach or DSU on each row/column.
    # Given H*W <= 4*10^5, we can store these in 1D arrays.
    
    # Initialize "next/prev" pointers for each cell.
    # For rows (left/right)
    row_right = [j + 1 for i in range(H) for j in range(W)]
    row_left = [j - 1 for i in range(H) for j in range(W)]
    # For cols (up/down)
    col_down = [i + 1 for i in range(H) for j in range(W)] # Simplified logic below
    
    # Actually, it's easier to manage 4 sets of DSU arrays.
    # For row r: left_dsu[r], right_dsu[r]
    # For col c: up_dsu[c], down_dsu[c]
    
    # To optimize memory and speed in Python:
    # Use 1D arrays to simulate 2D.
    # r_right[r * W + c] stores the index of the next potential wall to the right.
    r_right = [j + 1 for i in range(H) for j in range(W)]
    r_left = [j - 1 for i in range(H) for j in range(W)]
    c_down = [i + 1 for i in range(H) for j in range(W)] # This is wrong, needs to be (i+1)*W + j
    # Let's redefine.
    
    # To efficiently find the next wall, we use DSU.
    # For each row, two DSUs: one for right, one for left.
    # For each col, two DSUs: one for down, one for up.
    
    # Since H*W is small enough, let's use 1D arrays for the DSU parents.
    # parent_right[r*W + c] = next column index in row r that might have a wall.
    p_right = [j + 1 for i in range(H) for j in range(W)]
    p_left = [j - 1 for i in range(H) for j in range(W)]
    p_down = [(i + 1) * W + 0 for i in range(H) for j in range(W)] # Wrong
    # Correcting indexing:
    p_down = [(i + 1) * W + j if i < H - 1 else -1 for i in range(H) for j in range(W)]
    p_up = [(i - 1) * W + j if i > 0 else -1 for i in range(H) for j in range(W)]
    
    # To make it work like DSU (path compression):
    def find_right(r, c):
        idx = r * W + c
        if c >= W or p_right[idx] == c: return c
        # We need to store the actual column index. 
        # Let's use a different approach.
        pass

    # Revised strategy: use a simple 1D array for wall status and 
    # use DSU per row and per column.
    
    # Using a 1D array for wall status
    # wall_exists[r * W + c]
    
    # For each row, we have two DSUs to find the next existing wall.
    # For each col, we have two DSUs.
    # But H*W is 4e5, so 4 * 4e5 integers is fine.
    
    # Let's use a simpler approach: a set of existing walls per row and column.
    # But sets are slow. Let's use DSU.
    
    # For row r, walls are at columns c.
    # row_dsu_right[r][c] points to the next wall index >= c.
    # We can use 1D arrays:
    # right[r*W + c], left[r*W + c], down[r*W + c], up[r*W + c]
    
    # To implement DSU efficiently:
    def find(parent, i):
        root = i
        while root != -1 and parent[root] != root:
            root = parent[root]
        while i != -1 and parent[i] != root:
            next_i = parent[i]
            parent[i] = root
            i = next_i
        return root

    # This is still complex. Let's use a simpler approach.
    # For each row, we use a DSU to find the next available wall.
    # For each col, we use a DSU to find the next available wall.
    
    # Actually, the most efficient way in Python for this is 
    # using a list of lists or a flat array and manual path compression.
    
    # Re-initializing pointers for the 4 directions
    # Each element is the index of the next potential wall.
    # Row-wise (columns 0 to W-1)
    # col_right[r][c] = c; when wall at c is gone, col_right[r][c] = col_right[r][c+1]
    
    # We'll use 4 arrays of size H*W
    # right[r*W + c], left[r*W + c], down[r*W + c], up[r*W + c]
    # initialized to themselves.
    right = [j for i in range(H) for j in range(W)]
    left = [j for i in range(H) for j in range(W)]
    down = [i for i in range(H) for j in range(W)]
    up = [i for i in range(H) for j in range(W)]
    
    # To find the next wall:
    # def get_right(r, c):
    #     if c >= W: return W
    #     if wall_exists[r*W + c]: return c
    #     # ... DSU logic ...
    
    # Let's use a different DSU: 
    # For each row r, a DSU that merges c with c+1 when wall at c is destroyed.
    # Since we have 4 directions, 4 DSUs.
    
    # To avoid TLE in Python, we must be very efficient.
    # We'll use 4 arrays:
    # next_r[r*W + c], prev_r[r*W + c], next_c[r*W + c], prev_c[r*W + c]
    # initialized as:
    # next_r[r*W + c] = c + 1
    # prev_r[r*W + c] = c - 1
    # next_c[r*W + c] = i + 1
    # prev_c[r*W + c] = i - 1
    
    # But we only move the pointer when the wall is destroyed.
    
    # Let's use the property that we only destroy walls.
    # For each row, we maintain a DSU.
    # For each col, we maintain a DSU.
    
    # Let's use a simpler 2D-like structure with a 1D array.
    # wall[r*W + c] = 1 if wall exists, else 0.
    wall = [1] * (H * W)
    
    # DSU for each row (right), each row (left), each col (down), each col (up)
    # We use 4 arrays. For row r, the range is [r*W, r*W + W - 1]
    # For col c, the range is [c, c + W, c + 2W, ...]
    # This is getting confusing. Let's just use 4 arrays of size H*W.
    
    # next_col[r*W + c] is the next column in row r that might have a wall.
    # prev_col[r*W + c] is the previous column in row r that might have a wall.
    # next_row[r*W + c] is the next row in col c that might have a wall.
    # prev_row[r*W + c] is the previous row in col c that might have a wall.
    
    next_col = [j + 1 for i in range(H) for j in range(W)]
    prev_col = [j - 1 for i in range(H) for j in range(W)]
    next_row = [(i + 1) for i in range(H) for j in range(W)]
    prev_row = [(i - 1) for i in range(H) for j in range(W)]
    
    # DSU find functions
    def find_nc(r, c):
        if c < 0 or c >= W: return W
        idx = r * W + c
        if wall[idx]: return c
        next_col[idx] = find_nc(r, next_col[idx])
        return next_col[idx]

    def find_pc(r, c):
        if c < 0 or c >= W: return -1
        idx = r * W + c
        if wall[idx]: return c
        prev_col[idx] = find_pc(r, prev_col[idx])
        return prev_col[idx]

    def find_nr(r, c):
        if r < 0 or r >= H: return H
        idx = r * W + c
        if wall[idx]: return r
        next_row[idx] = find_nr(next_row[idx], c)
        return next_row[idx]

    def find_pr(r, c):
        if r < 0 or r >= H: return -1
        idx = r * W + c
        if wall[idx]: return r
        prev_row[idx] = find_pr(prev_row[idx], c)
        return prev_row[idx]

    # The recursive find will hit recursion limit. Let's use iterative.
    def find_nc_it(r, c):
        curr = c
        path = []
        while 0 <= curr < W and not wall[r * W + curr]:
            path.append(curr)
            curr = next_col[r * W + curr]
        for p in path: next_col[r * W + p] = curr
        return curr

    def find_pc_it(r, c):
        curr = c
        path = []
        while 0 <= curr < W and not wall[r * W + curr]:
            path.append(curr)
            curr = prev_col[r * W + curr]
        for p in path: prev_col[r * W + p] = curr
        return curr

    def find_nr_it(r, c):
        curr = r
        path = []
        while 0 <= curr < H and not wall[curr * W + c]:
            path.append(curr)
            curr = next_row[curr * W + c]
        for p in path: next_row[p * W + c] = curr
        return curr

    def find_pr_it(r, c):
        curr = r
        path = []
        while 0 <= curr < H and not wall[curr * W + c]:
            path.append(curr)