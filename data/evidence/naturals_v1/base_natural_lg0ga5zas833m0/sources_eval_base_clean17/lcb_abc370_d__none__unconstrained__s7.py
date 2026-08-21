import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To handle H*W up to 4e5 efficiently, we use sets or sorted lists to track existing walls.
    # However, since we need to find the "nearest" wall, a Disjoint Set Union (DSU) or 
    # a similar structure is needed for each row and column to skip empty cells.
    
    # parent_row[r][c] points to the next potential wall in row r
    # parent_col[c][r] points to the next potential wall in col c
    # We need 4 DSU structures per row/col to move Up, Down, Left, Right.
    
    # Instead of full DSU, we can use the fact that H*W is small.
    # We maintain sets of remaining walls for each row and each column.
    # But standard Python sets don't allow finding the nearest element.
    # We use a "linked list" approach via arrays to skip destroyed walls.
    
    # For each row, we maintain two arrays to find the next wall to the left and right.
    # For each col, we maintain two arrays to find the next wall above and below.
    
    # To save memory and time, we use a 1D array to represent the grid and 
    # 4 arrays for the "next" pointers.
    # L[r][c]: next wall index to the left of (r, c)
    # R[r][c]: next wall index to the right of (r, c)
    # U[r][c]: next wall index above (r, c)
    # D[r][c]: next wall index below (r, c)
    
    # Given constraints H*W <= 4e5, we can allocate these.
    # Using 1D arrays to simulate 2D for speed.
    
    # row_left[r][c], row_right[r][c], col_up[c][r], col_down[c][r]
    # We use a simple DSU-like path compression to skip empty cells.
    
    # Since Python is slow, we use a different approach: 
    # Store which cells are walls in a boolean array.
    # Use DSU to find the next existing wall.
    
    # To avoid MLE/TLE, let's use the property:
    # Each row i has a DSU for columns, and each col j has a DSU for rows.
    
    # row_next[r][c] = the index of the next wall in row r starting from c.
    # We need 4 such structures: Row-Right, Row-Left, Col-Down, Col-Up.
    
    # To implement this efficiently in Python:
    # For each row, we have two lists: right_ptr and left_ptr.
    # right_ptr[r][c] points to the next wall at or to the right of c.
    # left_ptr[r][c] points to the next wall at or to the left of c.
    
    # Pre-allocate
    # Using a flat array for the wall status
    is_wall = [True] * (H * W)
    
    # DSU structures for each row and column
    # row_r[r][c] is the next available wall in row r at index >= c
    row_r = [list(range(W + 1)) for _ in range(H)]
    row_l = [list(range(W + 2)) for _ in range(H)]
    col_d = [list(range(H + 1)) for _ in range(W)]
    col_u = [list(range(H + 2)) for _ in range(W)]
    
    def find_r(r, c):
        if row_r[r][c] == c: return c
        row_r[r][c] = find_r(r, row_r[r][c])
        return row_r[r][c]
    
    def find_l(r, c):
        if row_l[r][c] == c: return c
        row_l[r][c] = find_l(r, row_l[r][c])
        return row_l[r][c]
    
    def find_d(c, r):
        if col_d[c][r] == r: return r
        col_d[c][r] = find_d(c, col_d[c][r])
        return col_d[c][r]
    
    def find_u(c, r):
        if col_u[c][r] == r: return r
        col_u[c][r] = find_u(c, col_u[c][r])
        return col_u[c][r]

    # Correcting find functions to handle boundaries
    # row_r[r][c] stores the next wall index >= c. If no wall, it's W.
    # row_l[r][c] stores the next wall index <= c. If no wall, it's 0.
    # col_d[c][r] stores the next wall index >= r. If no wall, it's H.
    # col_u[c][r] stores the next wall index <= r. If no wall, it's 0.

    # Redefining find functions to avoid recursion depth issues
    def find_r_iter(r, c):
        root = c
        while root < W and row_r[r][root] != root:
            root = row_r[r][root]
        while c < W and row_r[r][c] != root:
            next_c = row_r[r][c]
            row_r[r][c] = root
            c = next_c
        return root

    def find_l_iter(r, c):
        root = c
        while root > 0 and row_l[r][root] != root:
            root = row_l[r][root]
        while c > 0 and row_l[r][c] != root:
            next_c = row_l[r][c]
            row_l[r][c] = root
            c = next_c
        return root

    def find_d_iter(c, r):
        root = r
        while root < H and col_d[c][root] != root:
            root = col_d[c][root]
        while r < H and col_d[c][r] != root:
            next_r = col_d[c][r]
            col_d[c][r] = root
            r = next_r
        return root

    def find_u_iter(c, r):
        root = r
        while root > 0 and col_u[c][root] != root:
            root = col_u[c][root]
        while r > 0 and col_u[c][r] != root:
            next_r = col_u[c][r]
            col_u[c][r] = root
            r = next_r
        return root

    def destroy(r, c):
        # r, c are 0-indexed
        if not is_wall[r * W + c]: return
        is_wall[r * W + c] = False
        # Update DSU pointers
        # In row r, cell c is no longer a wall
        row_r[r][c] = c + 1
        row_l[r][c] = c - 1
        # In col c, cell r is no longer a wall
        col_d[c][r] = r + 1
        col_u[c][r] = r - 1

    ptr = 3
    remaining_walls = H * W
    for _ in range(Q):
        rq = int(input_data[ptr]) - 1
        cq = int(input_data[ptr+1]) - 1
        ptr += 2
        
        if is_wall[rq * W + cq]:
            destroy(rq, cq)
            remaining_walls -= 1
        else:
            # Look Up
            res_u = find_u_iter(cq, rq - 1)
            if res_u >= 0:
                destroy(res_u, cq)
                remaining_walls -= 1
            # Look Down
            res_d = find_d_iter(cq, rq + 1)
            if res_d < H:
                destroy(res_d, cq)
                remaining_walls -= 1
            # Look Left
            res_l = find_l_iter(rq, cq - 1)
            if res_l >= 0:
                destroy(rq, res_l)
                remaining_walls -= 1
            # Look Right
            res_r = find_r_iter(rq, cq + 1)
            if res_r < W:
                destroy(rq, res_r)
                remaining_walls -= 1
                
    print(remaining_walls)

if __name__ == "__main__":
    solve()