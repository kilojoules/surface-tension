```python
import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Since H*W <= 4*10^5, we can't use a simple 2D array if H or W is very large.
    # However, we need to efficiently find the nearest wall in 4 directions.
    # We can use a Disjoint Set Union (DSU) or a similar structure to skip empty cells.
    # For each row, we maintain a DSU to find the next wall to the left and right.
    # For each column, we maintain a DSU to find the next wall above and below.
    
    # To save memory and handle the H*W constraint, we use a 1D representation or 
    # simply lists of lists if memory allows. H*W is 4*10^5, which is fine for Python.
    
    # walls[r][c] = True if there is a wall at (r, c)
    # Using 0-indexing internally
    walls = [[True] * W for _ in range(H)]
    
    # row_next[r][c] will help find the next existing wall in row r.
    # We use a DSU-like structure for each row and column to skip empty cells.
    # row_right[r][c] points to the next index >= c that might have a wall.
    # row_left[r][c] points to the next index <= c that might have a wall.
    
    # To implement this efficiently in Python, we can use a simple DSU per row/col.
    # However, given the constraints and the "first wall" logic, 
    # we can use a set of existing indices for each row and column.
    # But SortedList from sortedcontainers is not standard. 
    # We can use a DSU to find the nearest available wall.
    
    # Let's use DSU to track "empty" segments.
    # For each row r, row_up[r][c] and row_down[r][c]... wait, it's simpler:
    # For each row r, we need to find the nearest j < C_q and j > C_q where walls[r][j] is True.
    # For each col c, we need to find the nearest i < R_q and i > R_q where walls[i][c] is True.
    
    # Given the constraints, we can use a DSU for each row and each column.
    # row_right[r][c]: the index of the next wall to the right (including c)
    # row_left[r][c]: the index of the next wall to the left (including c)
    # col_down[c][r]: the index of the next wall below (including r)
    # col_up[c][r]: the index of the next wall above (including r)
    
    # To implement this without too much memory, we use 1D arrays for DSU.
    # But the number of rows/cols can be large.
    # Let's use the property: we only care about the "next" wall.
    # We can maintain for each row a DSU for 'right' and 'left' movements, 
    # and for each column a DSU for 'down' and 'up' movements.
    
    # Actually, a simpler way to implement "find next wall" is using a DSU 
    # where we union a cell with its neighbor once the wall is destroyed.
    
    # DSU structures
    # row_right[r][c] -> next possible wall index >= c
    # row_left[r][c] -> next possible wall index <= c
    # col_down[c][r] -> next possible wall index >= r
    # col_up[c][r] -> next possible wall index <= r
    
    # To optimize memory, we only create the DSU arrays when needed or use a flat array.
    # Since H*W is 4*10^5, we can afford 4 such arrays.
    
    # Using flat arrays to avoid list of lists overhead
    # index = r * W + c
    row_right = [i for i in range(H * W)]
    row_left = [i for i in range(H * W)]
    col_down = [i for i in range(H * W)]
    col_up = [i for i in range(H * W)]
    
    # Note: col_down/up indices are tricky. Let's use row-major for all.
    # col_down[r*W + c] will store the row index of the next wall.
    
    # Correcting DSU logic:
    # row_right[r][c] stores the index of the next wall in row r at or to the right of c.
    # We can use a simple DSU: when wall at (r, c) is destroyed, 
    # row_right[r][c] = row_right[r][c+1]
    
    # Re-initializing to be more memory efficient
    # Using 1D arrays for each row/col to avoid object overhead
    # We need 4 sets of DSUs.
    # For each row: right_dsu, left_dsu
    # For each col: down_dsu, up_dsu
    
    # Since H*W is small, we can use a simple 2D list for 'walls'
    # and 4 sets of DSUs.
    
    # To avoid TLE in Python, we use a fast DSU find.
    def find(parent, i, step):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            next_i = parent[i]
            parent[i] = root
            i = next_i
        return root

    # Because we need 4 directions, we'll use 4 separate arrays for the DSU parents.
    # To handle boundaries, we pad the grids.
    # Row DSUs: for each row r, parents for right (0..W) and left (0..W)
    # Col DSUs: for each col c, parents for down (0..H) and up (0..H)
    
    # To avoid memory issues and complexity, let's use a simpler approach:
    # For each row and col, maintain a DSU.
    # row_r[r][c] is the parent for finding the next wall to the right.
    # row_l[r][c] is the parent for finding the next wall to the left.
    # col_d[c][r] is the parent for finding the next wall down.
    # col_u[c][r] is the parent for finding the next wall up.
    
    # Using a single list for each direction to be faster.
    # row_r[r * (W+2) + c]
    # We add padding of 1 cell around the grid to avoid boundary checks.
    
    R_W = W + 2
    row_r = list(range(H * R_W)) # Simplified: we'll use a different indexing
    # Let's use a list of lists for simplicity, but be careful with memory.
    
    # Actually, the most efficient way in Python is to use 1D arrays and manual indexing.
    # row_r[r * (W+2) + c]
    # But H*W is 4*10^5, so 4 * 4*10^5 * 4 bytes is small.
    
    # Let's use a more Pythonic approach: 
    # a wall is destroyed -> we link it to its neighbor.
    
    # To handle the logic:
    # If wall at (r, c) exists: destroy it.
    # If not: destroy nearest walls in 4 directions.
    
    # To make this fast, we use a DSU for each row (left/right) and each col (up/down).
    # For row r:
    #   Right-DSU: if wall at (r, c) is gone, parent[c] = c + 1
    #   Left-DSU: if wall at (r, c) is gone, parent[c] = c - 1
    # For col c:
    #   Down-DSU: if wall at (r, c) is gone, parent[r] = r + 1
    #   Up-DSU: if wall at (r, c) is gone, parent[r] = r - 1
    
    # We need to store whether a wall exists.
    wall_exists = [True] * (H * W)
    
    # DSU parents
    # We use 4 arrays. To handle indexing, we can use a list of lists or flat.
    # Flat is usually faster.
    # row_r: H rows, each W+1 elements
    # row_l: H rows, each W+1 elements
    # col_d: W cols, each H+1 elements
    # col_u: W cols, each H+1 elements
    
    # But H and W can be unbalanced. Let's use a list of lists for the DSUs.
    row_r = [[i for i in range(W + 2)] for _ in range(H)]
    row_l = [[i for i in range(W + 2)] for _ in range(H)]
    col_d = [[i for i in range(H + 2)] for _ in range(W)]
    col_u = [[i for i in range(H + 2)] for _ in range(W)]
    
    def find_r(r, i):
        p = row_r[r]
        root = i
        while p[root] != root: root = p[root]
        while p[i] != root:
            nxt = p[i]
            p[i] = root
            i = nxt
        return root

    def find_l(r, i):
        p = row_l[r]
        root = i
        while p[root] != root: root = p[root]
        while p[i] != root:
            nxt = p[i]
            p[i] = root
            i = nxt
        return root

    def find_d(c, i):
        p = col_d[c]
        root = i
        while p[root] != root: root = p[root]
        while p[i] != root:
            nxt = p[i]
            p[i] = root
            i = nxt
        return root

    def find_u(c, i):
        p = col_u[c]
        root = i
        while p[root] != root: root = p[root]
        while p[i] != root:
            nxt = p[i]
            p[i] = root
            i = nxt
        return root

    def destroy(r, c):
        # r, c are 0-indexed
        if not wall_exists[r * W + c]:
            return
        wall_exists[r * W + c] = False
        # Update DSUs
        # row_r[r][c] = find_r(r, c + 1)
        row_r[r][c] = c + 1
        # row_l[r][c] = find_l(r, c - 1)
        row_l[r][c] = c - 1
        # col_d[c][r] = find_d(c, r + 1)
        col_d[c][r] = r + 1
        # col_u[c][r] = find_u(c, r - 1)
        col_u[c][r] = r - 1

    # Process queries
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr]) - 1
        cq = int(input_data[ptr+1]) - 1
        ptr += 2
        
        if wall_exists[rq * W + cq]:
            destroy(rq, cq)
        else:
            # Find 4 neighbors
            # Right
            res_c = find_r(rq, cq + 1)
            if res_c < W:
                destroy(rq, res_c)
            # Left
            res_c = find_l(rq, cq - 1)
            if res_c >= 0:
                destroy(rq, res_c)
            # Down
            res_r = find_d(cq, rq + 1)
            if res_r < H:
                destroy(res_r, cq)
            # Up
            res_r = find_u(cq, rq - 1)
            if res_r >= 0:
                destroy(