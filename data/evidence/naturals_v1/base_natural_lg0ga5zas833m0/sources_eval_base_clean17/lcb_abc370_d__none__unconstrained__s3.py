import sys

def solve():
    # Using fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To efficiently find the nearest wall, we use a Disjoint Set Union (DSU)
    # or similar structure. Since H*W <= 4e5, we can maintain a list of 
    # available walls for each row and each column.
    # However, a simpler way to implement "find nearest" is using 
    # linked-list style structures or DSU to skip empty cells.
    
    # We need to track if a wall exists at (r, c)
    # Using a flat array for memory efficiency: index = (r-1)*W + (c-1)
    walls = [True] * (H * W)
    
    # To find the nearest wall in a row or column, we use DSU.
    # For each row, we have two DSUs: one for finding the next wall to the right,
    # and one for the next wall to the left. Similarly for columns.
    
    # row_next[r][c] points to the next possible wall index >= c in row r
    # row_prev[r][c] points to the next possible wall index <= c in row r
    # col_next[c][r] points to the next possible wall index >= r in col c
    # col_prev[c][r] points to the next possible wall index <= r in col c
    
    # To save memory and avoid overhead, we can use a dictionary or a 1D array 
    # if H*W is small, but the constraints are 4e5.
    # Let's use a simpler approach: for each row and column, maintain a sorted 
    # list of existing walls. But deletions in sorted lists are O(N).
    # Instead, we use DSU to skip empty cells.
    
    parent_rn = [list(range(W + 2)) for _ in range(H)] # row next
    parent_rp = [list(range(W + 2)) for _ in range(H)] # row prev
    parent_cn = [list(range(H + 2)) for _ in range(W)] # col next
    parent_cp = [list(range(H + 2)) for _ in range(W)] # col prev

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
        # r, c are 0-indexed
        idx = r * W + c
        if not walls[idx]:
            return False
        walls[idx] = False
        
        # Update DSU for row
        # For next: current cell now points to the cell to its right
        parent_rn[r][c + 1] = find(parent_rn[r], c + 1) # This is not quite right for DSU
        # Let's redefine DSU: 
        # parent_rn[r][c] is the next wall index >= c.
        # When wall at c is removed, parent_rn[r][c] = find(parent_rn[r], c + 1)
        return True

    # Correct DSU logic for skipping
    # row_next[r][c] = find(row_next[r], c + 1)
    # row_prev[r][c] = find(row_prev[r], c - 1)
    
    # Re-initialize parents
    rn = [list(range(W + 2)) for _ in range(H)]
    rp = [list(range(W + 2)) for _ in range(H)]
    cn = [list(range(H + 2)) for _ in range(W)]
    cp = [list(range(H + 2)) for _ in range(W)]

    def find_n(p, i):
        if p[i] == i: return i
        p[i] = find_n(p, p[i])
        return p[i]

    def find_p(p, i):
        if p[i] == i: return i
        p[i] = find_p(p, p[i])
        return p[i]

    def destroy(r, c):
        # r, c are 0-indexed
        idx = r * W + c
        if walls[idx]:
            walls[idx] = False
            # Update DSUs to skip this cell
            rn[r][c] = find_n(rn[r], c + 1)
            rp[r][c] = find_p(rp[r], c - 1)
            cn[c][r] = find_n(cn[c], r + 1)
            cp[c][r] = find_p(cp[c], r - 1)
            return 1
        return 0

    total_destroyed = 0
    ptr = 3
    for _ in range(Q):
        rq = int(input_data[ptr]) - 1
        cq = int(input_data[ptr+1]) - 1
        ptr += 2
        
        if walls[rq * W + cq]:
            total_destroyed += destroy(rq, cq)
        else:
            # Look Up
            up = find_p(cp[cq], rq - 1)
            if up >= 0:
                total_destroyed += destroy(up, cq)
            # Look Down
            down = find_n(cn[cq], rq + 1)
            if down < H:
                total_destroyed += destroy(down, cq)
            # Look Left
            left = find_p(rp[rq], cq - 1)
            if left >= 0:
                total_destroyed += destroy(rq, left)
            # Look Right
            right = find_n(rn[rq], cq + 1)
            if right < W:
                total_destroyed += destroy(rq, right)

    print(H * W - total_destroyed)

# Increase recursion depth for DSU
sys.setrecursionlimit(1000000)
solve()