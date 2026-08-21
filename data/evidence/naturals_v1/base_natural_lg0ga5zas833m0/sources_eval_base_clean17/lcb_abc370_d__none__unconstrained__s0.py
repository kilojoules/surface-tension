import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # The constraints H*W <= 4*10^5 and Q <= 2*10^5 suggest an efficient 
    # way to find the nearest existing wall.
    # Since we need to find the nearest wall in 4 directions, we can use 
    # a Disjoint Set Union (DSU) or a similar structure to skip empty cells.
    
    # To handle the grid efficiently, we create DSU structures for each row and each column.
    # row_dsu[r] will manage the connectivity of empty cells in row r.
    # col_dsu[c] will manage the connectivity of empty cells in column c.
    
    # However, a simpler way to implement "find the next wall" is to maintain 
    # a set of active walls or use a DSU that points to the next available wall.
    
    # Let's use a DSU-like approach: for each row, we maintain two DSUs 
    # (one for moving left, one for moving right). Same for columns.
    
    # Since H*W is small, we can represent the grid as a 1D array or a list of lists.
    # wall_exists[r][c] = True/False
    # To avoid memory overhead and handle H, W flexibly, we use a 1D array.
    wall_exists = [True] * (H * W)
    
    # DSU for rows: next_right[r][c] and next_left[r][c]
    # DSU for cols: next_down[c][r] and next_up[c][r]
    
    # To keep it memory efficient and avoid TLE in Python, we can use a 
    # different approach: maintain sorted lists of existing walls per row/col
    # but that's O(Q * log(W)) which might be slow if we remove elements.
    # Instead, we use a DSU to skip deleted cells.
    
    # For each row r:
    # right[r][j] points to the next index k >= j that might have a wall.
    # left[r][j] points to the next index k <= j that might have a wall.
    
    # To implement this efficiently in Python:
    # We use 4 arrays of parents for the DSU.
    # row_r[r * (W+2) + c], row_l[r * (W+2) + c]
    # col_d[c * (H+2) + r], col_u[c * (H+2) + r]
    
    # Actually, a simpler way is to use a dictionary or a list of sets,
    # but Python's `bisect` on sorted lists is very fast.
    # However, removing from a list is O(N).
    # Let's use the DSU approach with a flat array for speed.
    
    # Given H*W <= 4*10^5, we can afford a few such arrays.
    # We'll use 1-based indexing for R, C to match input.
    
    # row_next[r][c] -> next available wall index in row r starting from c
    # We use 4 parent arrays to find the nearest wall in 4 directions.
    # For row r: 
    #   up_p[c][r] stores the next wall above (r, c)
    #   down_p[c][r] stores the next wall below (r, c)
    #   left_p[r][c] stores the next wall to the left of (r, c)
    #   right_p[r][c] stores the next wall to the right of (r, c)
    
    # To optimize memory and time in Python, we use a simpler structure.
    # We can use `set` and `bisect` if we use a SortedList from sortedcontainers, 
    # but that's not standard. 
    # Standard Python `bisect` on a list is fast, but `pop` is slow.
    # Let's use DSU.
    
    parent_right = [list(range(W + 2)) for _ in range(H + 1)]
    parent_left = [list(range(W + 2)) for _ in range(H + 1)]
    parent_down = [list(range(H + 2)) for _ in range(W + 1)]
    parent_up = [list(range(H + 2)) for _ in range(W + 1)]
    
    def find(p, i):
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            next_i = p[i]
            p[i] = root
            i = next_i
        return root

    def remove_wall(r, c):
        # Mark wall as gone
        wall_exists[(r-1)*W + (c-1)] = False
        # Update DSUs: point to the neighbor
        # Row DSU
        parent_right[r][c] = find(parent_right[r], c + 1)
        parent_left[r][c] = find(parent_left[r], c - 1)
        # Col DSU
        parent_down[c][r] = find(parent_down[c], r + 1)
        parent_up[c][r] = find(parent_up[c], r - 1)

    total_walls = H * W
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr])
        cq = int(input_data[ptr+1])
        ptr += 2
        
        if wall_exists[(rq-1)*W + (cq-1)]:
            remove_wall(rq, cq)
            total_walls -= 1
        else:
            # Look Up
            res_r = find(parent_up[cq], rq - 1)
            if res_r >= 1:
                remove_wall(res_r, cq)
                total_walls -= 1
            # Look Down
            res_r = find(parent_down[cq], rq + 1)
            if res_r <= H:
                remove_wall(res_r, cq)
                total_walls -= 1
            # Look Left
            res_c = find(parent_left[rq], cq - 1)
            if res_c >= 1:
                remove_wall(rq, res_c)
                total_walls -= 1
            # Look Right
            res_c = find(parent_right[rq], cq + 1)
            if res_c <= W:
                remove_wall(rq, res_c)
                total_walls -= 1
                
    print(total_walls)

if __name__ == "__main__":
    solve()