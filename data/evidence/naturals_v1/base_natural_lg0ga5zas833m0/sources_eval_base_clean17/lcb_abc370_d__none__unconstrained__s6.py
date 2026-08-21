```python
import sys

def solve():
    # Increase recursion depth just in case, although not needed for this iterative approach
    sys.setrecursionlimit(10**6)
    
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To handle the walls efficiently, we need to find the nearest wall in 4 directions.
    # Since H*W is up to 4*10^5, we can't use a simple 2D array for searching.
    # We use a DSU-like structure or a sorted list (via bisect) to find the nearest remaining walls.
    # However, since we need to search in both rows and columns, we maintain sets of remaining wall indices.
    
    # To save memory and time, we use lists of sets or similar. 
    # But Python sets/bisect on lists might be slow. 
    # A better way is to use a Linked List approach or a DSU to skip empty cells.
    
    # Let's use a DSU-like structure for each row and each column.
    # row_up[r][c] points to the next available wall index <= c
    # row_down[r][c] points to the next available wall index >= c
    # col_up[c][r] points to the next available wall index <= r
    # col_down[c][r] points to the next available wall index >= r
    
    # Given the constraints H*W <= 4*10^5, we can flatten the 2D grid.
    # But it's easier to maintain 4 arrays of "next" pointers for each row/col.
    
    # To implement the "find nearest wall" efficiently:
    # For each row, we maintain a way to find the nearest index j < C and j > C.
    # For each col, we maintain a way to find the nearest index i < R and i > R.
    
    # Using lists and updating them (similar to a linked list) is efficient.
    # L[r][c] = index of wall to the left of (r, c)
    # R[r][c] = index of wall to the right of (r, c)
    # U[r][c] = index of wall above (r, c)
    # D[r][c] = index of wall below (r, c)
    
    # We initialize these pointers.
    # Since H*W is large, we use 1D arrays to simulate 2D to avoid overhead.
    # index = (r-1)*W + (c-1)
    
    # However, a simpler approach: use a boolean array for walls and a DSU 
    # to find the next 'True' value. But we need 4 DSUs per row/col.
    
    # Let's use the "linked list" approach. 
    # For each row r: left[c], right[c]
    # For each col c: up[r], down[r]
    
    # We can't afford 4 * H * W integers if we are not careful with memory.
    # H*W = 4e5. 4 * 4e5 * 8 bytes approx 12.8 MB. Python integers are larger.
    # We'll use arrays.
    
    import array
    
    # Row pointers
    # row_left[r*W + c] is the column index of the wall to the left of c in row r.
    # row_right[r*W + c] is the column index of the wall to the right of c in row r.
    # col_up[c*H + r] is the row index of the wall above r in col c.
    # col_down[c*H + r] is the row index of the wall below r in col c.
    
    # Using 0-based indexing internally
    # row_left[r][c] = c-1, row_right[r][c] = c+1
    # col_up[c][r] = r-1, col_down[c][r] = r+1
    
    # To handle boundaries, we use 0 and W+1 / H+1.
    
    # Since H*W is the bottleneck, we use a flat array for the wall status.
    exists = [True] * (H * W)
    
    # To find the nearest wall, we can use a DSU-like jump pointer.
    # For each row:
    # row_prev[r][c] = the column of the wall to the left of c (inclusive)
    # row_next[r][c] = the column of the wall to the right of c (inclusive)
    # Similarly for cols.
    
    # Because we need to update and query, we can use the property that once a wall is gone,
    # it's gone forever. We can use a DSU to skip empty cells.
    
    # Actually, the most efficient way in Python for this is to use a 
    # dictionary of sets or just maintain the walls in a sorted list per row/col.
    # But with 4e5 cells, sorted lists (bisect) might be too slow.
    
    # Let's use the "jump" array (DSU) approach.
    # For each row, we have two DSUs: one for finding the next wall to the right, one for the left.
    
    # However, a simpler way:
    # Maintain the walls as a set of indices for each row and each column.
    # row_walls[r] = {c1, c2, ...}
    # col_walls[c] = {r1, r2, ...}
    # This is still slow. 
    
    # Let's use the linked-list approach with arrays.
    # For each row, we store the 'next' and 'prev' wall indices.
    # When a wall at (r, c) is destroyed:
    # row_next[r][prev[r][c]] = row_next[r][c]
    # row_prev[r][next[r][c]] = row_prev[r][c]
    # (and similarly for columns)
    
    # To implement this, we need to handle the 2D grid as 1D.
    # row_prev[r*W + c], row_next[r*W + c], col_prev[c*H + r], col_next[c*H + r]
    
    # Using array.array for memory efficiency
    # 'i' is signed int (usually 4 bytes)
    row_prev = array.array('i', [0] * (H * W))
    row_next = array.array('i', [0] * (H * W))
    col_prev = array.array('i', [0] * (H * W))
    col_next = array.array('i', [0] * (H * W))
    
    for r in range(H):
        for c in range(W):
            idx = r * W + c
            row_prev[idx] = c - 1
            row_next[idx] = c + 1
            col_prev[idx] = r - 1
            col_next[idx] = r + 1
            
    # To handle the boundaries (c-1 = -1, c+1 = W), we can just check bounds.
    
    # The problem is that row_prev[idx] needs to be the column of the actual wall.
    # The linked list should be:
    # For each row r: a list of nodes where each node is a column index.
    # When wall at (r, c) is destroyed, we remove c from the linked list of row r.
    
    # Let's use a different approach: 4 arrays of "jump" pointers.
    # For each row r: 
    # L[r][c] = the index of the wall to the left of c.
    # R[r][c] = the index of the wall to the right of c.
    # Initially L[r][c] = c-1, R[r][c] = c+1.
    # When wall at (r, c) is destroyed, we don't just update L and R for (r, c).
    # We need to update the pointers of the *neighbors*.
    
    # Correct Linked List logic:
    # For each row r, we have a linked list of existing wall columns.
    # For each col c, we have a linked list of existing wall rows.
    # When wall at (r, c) is destroyed:
    # 1. Find the wall to the left (L_c) and right (R_c) in row r.
    # 2. Set R[r][L_c] = R_c and L[r][R_c] = L_c.
    # 3. Similarly for columns.
    
    # We need to store L, R, U, D for every cell.
    # L[r][c] = column index of the wall to the left.
    # R[r][c] = column index of the wall to the right.
    # U[r][c] = row index of the wall above.
    # D[r][c] = row index of the wall below.
    
    # Memory: 4 * H * W * 4 bytes = 16 * 4e5 = 6.4 MB. This is fine.
    # We use 1D arrays to represent 2D.
    
    # To handle boundaries, we can use a dummy wall at index -1 and W.
    L = array.array('i', [0] * (H * W))
    R = array.array('i', [0] * (H * W))
    U = array.array('i', [0] * (H * W))
    D = array.array('i', [0] * (H * W))
    
    for r in range(H):
        for c in range(W):
            idx = r * W + c
            L[idx] = c - 1
            R[idx] = c + 1
            U[idx] = r - 1
            D[idx] = r + 1
            
    wall_exists = [True] * (H * W)
    remaining_walls = H * W
    
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr]) - 1
        cq = int(input_data[ptr+1]) - 1
        ptr += 2
        
        idx = rq * W + cq
        if wall_exists[idx]:
            # Destroy wall at (rq, cq)
            wall_exists[idx] = False
            remaining_walls -= 1
            
            # Update neighbors in row
            l_col = L[idx]
            r_col = R[idx]
            if l_col >= 0:
                R[rq * W + l_col] = r_col
            if r_col < W:
                L[rq * W + r_col] = l_col
                
            # Update neighbors in col
            u_row = U[idx]
            d_row = D[idx]
            if u_row >= 0:
                D[u_row * W + cq] = d_row
            if d_row < H:
                U[d_row * W + cq] = u_row
                
        else:
            # No wall at (rq, cq), destroy nearest walls in 4 directions
            # Looking Up
            # Since (rq, cq) is no longer a wall, we need to know where the nearest walls are.
            # We can store the current "links" for the empty cell too.
            # Wait, the L, R, U, D arrays for an empty cell (rq, cq) 
            # will actually store the current nearest walls because they were updated 
            # when the wall at (rq, cq) was first destroyed.
            
            # Let's check:
            # When (rq, cq) was destroyed, we updated L[rq][R[rq][cq]] = L[rq][cq]
            # and R[rq][L[rq][cq]] = R[rq][cq].
            # But we didn't update L[rq][cq] and R[rq][cq] themselves.
            # They still point to the neighbors at the moment of destruction.
            # To keep them updated, when we destroy a wall at (r, c), 
            # we should update the L, R, U, D of the empty cell (r, c) too.
            
            # Actually, the simplest way to keep the empty cell (rq, cq) updated
            # is to realize that L[rq][cq] and R[rq][cq] are only updated 
            # when the walls to their immediate left/right are destroyed.
            # So we just need to