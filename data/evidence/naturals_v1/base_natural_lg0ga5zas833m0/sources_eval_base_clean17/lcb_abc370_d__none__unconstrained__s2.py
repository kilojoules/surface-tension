import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To handle the grid efficiently, we use a Disjoint Set Union (DSU) 
    # or similar structure to skip empty cells.
    # Since H*W is up to 4*10^5, we can represent the grid as a 1D array.
    # For each row, we maintain a DSU for columns, and for each column, a DSU for rows.
    
    # row_dsu[r] stores connectivity for cells in row r (columns)
    # col_dsu[c] stores connectivity for cells in column c (rows)
    # However, a simpler way is to use two sets of DSUs: 
    # one to find the next wall to the right, one to the left, etc.
    
    # For each row: 
    # right[r][c] points to the next possible wall at index >= c
    # left[r][c] points to the next possible wall at index <= c
    # Similarly for columns.
    
    # To save memory and avoid overhead, we use a flat array and a custom DSU.
    # But since we need 4 directions, we need 4 sets of "next" pointers.
    
    # Using a simple 2D array for 'wall existence'
    # and 4 arrays of DSUs to find the nearest walls.
    # Memory limit is usually generous, but H*W is 4*10^5.
    
    walls = [True] * (H * W)
    
    # DSU structures to find the next wall
    # row_next[r][c] -> next wall in row r at index >= c
    # row_prev[r][c] -> next wall in row r at index <= c
    # col_next[c][r] -> next wall in col c at index >= r
    # col_prev[c][r] -> next wall in col c at index <= r
    
    # To implement this efficiently in Python, we use parent arrays.
    # parent_rn[r * (W+2) + c]
    
    rn = [i for i in range(H * (W + 2))]
    rl = [i for i in range(H * (W + 2))]
    cn = [i for i in range(W * (H + 2))]
    cp = [i for i in range(W * (H + 2))]
    
    # We need to map (r, c) to a unique index. 
    # r: 0 to H-1, c: 0 to W-1
    # To handle boundaries, we treat the grid as 1-indexed for the DSU logic
    # but 0-indexed for the wall array.
    
    def find(parent, i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            next_i = parent[i]
            parent[i] = root
            i = next_i
        return root

    # Instead of 4 DSUs per cell, let's use a simpler approach.
    # Each row has a DSU for 'next wall' and 'prev wall'.
    # Each col has a DSU for 'next wall' and 'prev wall'.
    
    # row_next[r][c] is the next wall in row r at col >= c
    # We can use 1D arrays to represent these.
    # row_next[r * (W+2) + c]
    
    # Initialize parents for boundaries
    # For row_next: parent[i] = i. When wall at (r, c) is gone, parent[c] = c + 1
    # For row_prev: parent[i] = i. When wall at (r, c) is gone, parent[c] = c - 1
    
    # Re-initializing to match 1-based indexing for queries
    rn = [i for i in range(H * (W + 2))]
    rl = [i for i in range(H * (W + 2))]
    cn = [i for i in range(W * (H + 2))]
    cp = [i for i in range(W * (H + 2))]
    
    # We'll use a helper to get the index
    # row_next: (r-1) * (W+2) + c
    
    def get_rn(r, c): return (r-1) * (W+2) + c
    def get_rl(r, c): return (r-1) * (W+2) + c
    def get_cn(r, c): return (c-1) * (H+2) + r
    def get_cp(r, c): return (c-1) * (H+2) + r

    # To avoid TLE in Python, we use a more optimized DSU and iterative find.
    def find_rn(r, c):
        idx = get_rn(r, c)
        root = idx
        while rn[root] != root: root = rn[root]
        while rn[idx] != root:
            nxt = rn[idx]
            rn[idx] = root
            idx = nxt
        return root

    def find_rl(r, c):
        idx = get_rl(r, c)
        root = idx
        while rl[root] != root: root = rl[root]
        while rl[idx] != root:
            nxt = rl[idx]
            rl[idx] = root
            idx = nxt
        return root

    def find_cn(r, c):
        idx = get_cn(r, c)
        root = idx
        while cn[root] != root: root = cn[root]
        while cn[idx] != root:
            nxt = cn[idx]
            cn[idx] = root
            idx = nxt
        return root

    def find_cp(r, c):
        idx = get_cp(r, c)
        root = idx
        while cp[root] != root: root = cp[root]
        while cp[idx] != root:
            nxt = cp[idx]
            cp[idx] = root
            idx = nxt
        return root

    # To avoid the overhead of function calls in Python, 
    # we'll inline the logic and use a flat wall array.
    
    wall_exists = [True] * (H * W)
    remaining_walls = H * W
    
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr])
        cq = int(input_data[ptr+1])
        ptr += 2
        
        if wall_exists[(rq-1)*W + (cq-1)]:
            # Destroy wall at (rq, cq)
            wall_exists[(rq-1)*W + (cq-1)] = False
            remaining_walls -= 1
            # Update DSUs
            # Row next/prev
            rn[get_rn(rq, cq)] = find_rn(rq, cq + 1)
            rl[get_rl(rq, cq)] = find_rl(rq, cq - 1)
            # Col next/prev
            cn[get_cn(rq, cq)] = find_cn(rq + 1, cq)
            cp[get_cp(rq, cq)] = find_cp(rq - 1, cq)
        else:
            # Destroy first walls in 4 directions
            # Up
            up_idx = find_cp(rq, cq)
            # find_cp returns the index in the flat array, we need the actual row
            # The DSU for columns is stored as (c-1)*(H+2) + r
            # So row = root - (c-1)*(H+2)
            r_up = up_idx - (cq-1)*(H+2)
            if 1 <= r_up <= H:
                # Destroy wall at (r_up, cq)
                wall_exists[(r_up-1)*W + (cq-1)] = False
                remaining_walls -= 1
                rn[get_rn(r_up, cq)] = find_rn(r_up, cq + 1)
                rl[get_rl(r_up, cq)] = find_rl(r_up, cq - 1)
                cn[get_cn(r_up, cq)] = find_cn(r_up + 1, cq)
                cp[get_cp(r_up, cq)] = find_cp(r_up - 1, cq)
            
            # Down
            down_idx = find_cn(rq, cq)
            r_down = down_idx - (cq-1)*(H+2)
            if 1 <= r_down <= H:
                wall_exists[(r_down-1)*W + (cq-1)] = False
                remaining_walls -= 1
                rn[get_rn(r_down, cq)] = find_rn(r_down, cq + 1)
                rl[get_rl(r_down, cq)] = find_rl(r_down, cq - 1)
                cn[get_cn(r_down, cq)] = find_cn(r_down + 1, cq)
                cp[get_cp(r_down, cq)] = find_cp(r_down - 1, cq)
                
            # Left
            left_idx = find_rl(rq, cq)
            c_left = left_idx - (rq-1)*(W+2)
            if 1 <= c_left <= W:
                wall_exists[(rq-1)*W + (c_left-1)] = False
                remaining_walls -= 1
                rn[get_rn(rq, c_left)] = find_rn(rq, c_left + 1)
                rl[get_rl(rq, c_left)] = find_rl(rq, c_left - 1)
                cn[get_cn(rq, c_left)] = find_cn(rq + 1, c_left)
                cp[get_cp(rq, c_left)] = find_cp(rq - 1, c_left)
                
            # Right
            right_idx = find_rn(rq, cq)
            c_right = right_idx - (rq-1)*(W+2)
            if 1 <= c_right <= W:
                wall_exists[(rq-1)*W + (c_right-1)] = False
                remaining_walls -= 1
                rn[get_rn(rq, c_right)] = find_rn(rq, c_right + 1)
                rl[get_rl(rq, c_right)] = find_rl(rq, c_right - 1)
                cn[get_cn(rq, c_right)] = find_cn(rq + 1, c_right)
                cp[get_cp(rq, c_right)] = find_cp(rq - 1, c_right)

    print(remaining_walls)

if __name__ == "__main__":
    solve()