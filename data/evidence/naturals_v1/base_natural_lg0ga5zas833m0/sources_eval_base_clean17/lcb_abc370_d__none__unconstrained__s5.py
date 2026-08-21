import sys

def solve():
    # Increase recursion depth for safety, though not explicitly needed for this approach
    sys.setrecursionlimit(10**6)
    
    # Fast I/O reading
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To handle the constraints H*W <= 4e5, we use a flat array or a list of sets.
    # The problem requires finding the "first wall" in 4 directions.
    # This is a classic problem that can be solved using a Disjoint Set Union (DSU) 
    # or by maintaining sorted lists of existing walls for each row and column.
    # Given the constraints and the nature of the problem, 
    # using a sorted list (or a balanced BST) for each row and column is efficient.
    
    # However, in Python, we can use the 'bisect' module with sorted lists.
    # For each row, we maintain a sorted list of column indices that still have walls.
    # For each column, we maintain a sorted list of row indices that still have walls.
    
    import bisect
    
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # wall_exists[r][c] would be too large if H or W is very large, 
    # but H*W is limited. We can use a set of tuples or a bitset.
    # Using a set of tuples (r, c) to track remaining walls.
    # But since we need to count remaining walls, we can just count total destroyed.
    
    # Using a 1D array to track wall existence to save memory and time
    # index = (r-1)*W + (c-1)
    exists = [True] * (H * W)
    destroyed_count = 0
    
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr])
        cq = int(input_data[ptr+1])
        ptr += 2
        
        # Adjust to 0-indexed
        r_idx = rq - 1
        c_idx = cq - 1
        
        if exists[r_idx * W + c_idx]:
            # Destroy the wall at (rq, cq)
            exists[r_idx * W + c_idx] = False
            destroyed_count += 1
            
            # Remove from sorted lists
            # Using bisect to find the index to remove
            # Note: list.pop(i) is O(N). With H*W=4e5, this might be slow.
            # But we only remove each wall once. The bottleneck is the search.
            # To optimize, we can't use list.pop for 4e5 elements frequently.
            # Let's reconsider: DSU is better for "find next available".
            pass

# The list.pop approach is O(Q * (H or W)), which is too slow.
# Let's implement the solution using DSU for each row and column.
# We need 4 DSU structures per row/col: one for Up, Down, Left, Right.

def solve_dsu():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data: return
    H, W, Q = map(int, input_data[:3])
    
    # parent[0][r][c] -> next wall to the left
    # parent[1][r][c] -> next wall to the right
    # parent[2][r][c] -> next wall up
    # parent[3][r][c] -> next wall down
    
    # To save memory, we use 1D arrays for DSU.
    # row_left[r][c], row_right[r][c], col_up[r][c], col_down[r][c]
    # Each is a list of size W or H.
    
    row_l = [list(range(W + 2)) for _ in range(H)]
    row_r = [list(range(W + 2)) for _ in range(H)]
    col_u = [list(range(H + 2)) for _ in range(W)]
    col_d = [list(range(H + 2)) for _ in range(W)]
    
    def find(p, i):
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            next_i = p[i]
            p[i] = root
            i = next_i
        return root

    exists = [True] * (H * W)
    destroyed = 0
    
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr])
        cq = int(input_data[ptr+1])
        ptr += 2
        
        if exists[(rq-1)*W + (cq-1)]:
            exists[(rq-1)*W + (cq-1)] = False
            destroyed += 1
            # Update DSU: this cell no longer has a wall
            # For row rq-1, cell cq is gone.
            # Left DSU: cq points to cq-1
            row_l[rq-1][cq] = cq - 1
            # Right DSU: cq points to cq+1
            row_r[rq-1][cq] = cq + 1
            # Up DSU: rq points to rq-1
            col_u[cq-1][rq] = rq - 1
            # Down DSU: rq points to rq+1
            col_d[cq-1][rq] = rq + 1
        else:
            # Destroy 4 neighbors
            # Left
            target_c = find(row_l[rq-1], cq)
            if target_c >= 1:
                exists[(rq-1)*W + (target_c-1)] = False
                destroyed += 1
                row_l[rq-1][target_c] = target_c - 1
                row_r[rq-1][target_c] = target_c + 1
                col_u[target_c-1][rq] = rq - 1
                col_d[target_c-1][rq] = rq + 1
            
            # Right
            target_c = find(row_r[rq-1], cq)
            if target_c <= W:
                if exists[(rq-1)*W + (target_c-1)]:
                    exists[(rq-1)*W + (target_c-1)] = False
                    destroyed += 1
                    row_l[rq-1][target_c] = target_c - 1
                    row_r[rq-1][target_c] = target_c + 1
                    col_u[target_c-1][rq] = rq - 1
                    col_d[target_c-1][rq] = rq + 1
            
            # Up
            target_r = find(col_u[cq-1], rq)
            if target_r >= 1:
                if exists[(target_r-1)*W + (cq-1)]:
                    exists[(target_r-1)*W + (cq-1)] = False
                    destroyed += 1
                    row_l[target_r-1][cq] = cq - 1
                    row_r[target_r-1][cq] = cq + 1
                    col_u[cq-1][target_r] = target_r - 1
                    col_d[cq-1][target_r] = target_r + 1
            
            # Down
            target_r = find(col_d[cq-1], rq)
            if target_r <= H:
                if exists[(target_r-1)*W + (cq-1)]:
                    exists[(target_r-1)*W + (cq-1)] = False
                    destroyed += 1
                    row_l[target_r-1][cq] = cq - 1
                    row_r[target_r-1][cq] = cq + 1
                    col_u[cq-1][target_r] = target_r - 1
                    col_d[cq-1][target_r] = target_r + 1
                    
    print(H * W - destroyed)

if __name__ == "__main__":
    solve_dsu()