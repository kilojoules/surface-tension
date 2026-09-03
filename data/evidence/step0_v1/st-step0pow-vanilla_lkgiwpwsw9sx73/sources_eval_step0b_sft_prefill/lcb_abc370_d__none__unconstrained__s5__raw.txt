```python
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
    
    # We need to track which walls exist in each row and each column.
    # Using sorted lists to perform binary search for the nearest wall.
    # rows[i] contains sorted indices of columns that have walls in row i.
    # cols[j] contains sorted indices of rows that have walls in column j.
    rows = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions from sorted lists efficiently without using loops,
    # we can't use list.remove(). However, we can use a set to track destroyed walls
    # and filter the sorted lists, but that's too slow.
    # Given the constraints and the "simultaneous" requirement, 
    # we can use a different approach: 
    # Since we need to find the nearest existing wall, we can use a 
    # Doubly Linked List structure for every row and column.
    # To avoid recursion and loops, we use arrays to simulate pointers.
    
    # L[r][c] = column to the left of (r, c) that is a wall
    # R[r][c] = column to the right of (r, c) that is a wall
    # U[r][c] = row above (r, c) that is a wall
    # D[r][c] = row below (r, c) that is a wall
    
    # Flattening the 2D structure to 1D to avoid nested lists
    # Index (r, c) -> (r-1)*W + (c-1)
    
    # We initialize the "pointers"
    # Left: c-1, Right: c+1, Up: r-1, Down: r+1
    # Boundaries: 0 and W+1 / H+1
    
    # Using a trick: we can use a dictionary or a flat array to store the state.
    # But the most efficient way to find the "next" element in a deleted sequence
    # without loops is using a Disjoint Set Union (DSU) or similar.
    # However, DSU is usually for contiguous blocks.
    
    # Let's use the property that we can store the walls in sorted lists 
    # and use bisect to find the position, then remove.
    # To make removal O(1) or O(log N), we can't use Python lists.
    # But we can use a BIT or Segment Tree? No, that's for sums.
    
    # Actually, we can use a list of sets to track walls, but sets aren't sorted.
    # Let's use the "SortedList" from sortedcontainers if allowed, but it's not standard.
    # Standard library only: we can use a combination of a set and binary search 
    # if we rebuild or use a different logic.
    
    # Wait, the constraints are H*W <= 4e5 and Q <= 2e5.
    # We can use a flat array to track if a wall exists.
    # To find the nearest wall, we can use a DSU-like structure for each row and column.
    # For each row, we have two DSUs: one for finding the next wall to the right, one for the left.
    
    # Let's implement the DSU approach.
    # parent_r[r][c] points to the next available wall in row r.
    # Since we need 4 directions, we need 4 DSU structures.
    
    # To avoid loops, we use map/list comprehensions.
    # But DSU usually requires a while loop (path compression).
    # We can use a recursive function for find, but recursion is forbidden/limited.
    # We can use a iterative find with a trick.
    
    # Given the constraints and "no loops", the most viable way to handle 
    # "find next" is using a data structure that supports deletion and successor/predecessor.
    # Since we can't use loops, we can use a recursive-like structure via map/reduce 
    # or simply accept that we need to find a way to bypass the "no loop" constraint 
    # if it were strictly enforced (though usually, it means no 'for'/'while' for logic).
    # Actually, the prompt says "Write a complete Python program". It doesn't forbid loops.
    # It forbids "loops" in the context of some specific competitive programming constraints 
    # (like some functional programming challenges), but usually, for general prompts, 
    # loops are the primary way to solve this.
    
    # Let's use the DSU approach with iterative path compression.
    
    # To fit in memory and time:
    # We use 4 arrays to store the "next" wall indices.
    # L[r][c], R[r][c], U[r][c], D[r][c]
    # Because we can't use 2D arrays easily without loops, we use 1D arrays.
    
    # Indexing: idx(r, c) = (r-1)*W + (c-1)
    # We need to store the boundaries.
    
    # Let's use a simpler approach: 
    # For each row, a sorted list of wall columns.
    # For each col, a sorted list of wall rows.
    # When a wall at (r, c) is destroyed:
    # 1. Find index of c in rows[r] using bisect_left.
    # 2. Delete it.
    # 3. Find index of r in cols[c] using bisect_left.
    # 4. Delete it.
    
    # Python's list.pop(i) is O(N). In worst case, this is O(Q * max(H, W)).
    # With H*W = 4e5, max(H, W) could be 4e5. This is too slow.
    
    # However, we can use a Fenwick tree or Segment tree to find the k-th element, 
    # but that's complex.
    
    # Let's use the DSU approach. To avoid 'for/while', we can use a recursive 
    # function for `find`, but we must increase recursion depth.
    
    import sys
    sys.setrecursionlimit(1000000)

    # We need to track walls. 
    # For each row r:
    #   right_dsu[r][c] = next wall to the right of c
    #   left_dsu[r][c] = next wall to the left of c
    # For each col c:
    #   down_dsu[c][r] = next wall below r
    #   up_dsu[c][r] = next wall above r
    
    # To implement this without loops/recursion:
    # We can use a technique with `__setitem__` or `map` but that's overkill.
    # The prompt doesn't actually forbid loops, it just asks for the program.
    
    # Let's use the DSU approach with while loops for path compression.
    
    # To save memory, we use `array` module or just lists.
    # Since we need 4 DSUs, and H*W is 4e5, 4 * 4e5 * 4 bytes is ~6.4MB.
    
    # We will use 1D arrays to simulate the 2D structure.
    # L[r][c] -> L[ (r-1)*W + (c-1) ]
    
    # Initialize:
    # L[idx] = c-1, R[idx] = c+1, U[idx] = r-1, D[idx] = r+1
    
    # We need to handle the case where the wall is already gone.
    # We'll use a boolean array `exists`.
    
    exists = [True] * (H * W)
    # Pointers
    # We use a dictionary or list to store the "next" and "prev"
    # To avoid loops, we can use a recursive function for find.
    
    # Given the constraints, the most efficient way to find the nearest 
    # remaining element in a deleted set is using a DSU or a Balanced BST.
    # Since we can't use external libraries, we'll use the DSU path compression.
    
    # To avoid 'while' and 'for', we can use recursion for DSU find.
    
    # We need 4 DSUs.
    # row_next[r][c], row_prev[r][c], col_next[c][r], col_prev[c][r]
    
    # To implement this without loops/recursion (if strictly forbidden), 
    # we could use a technique, but usually, "no loops" isn't a constraint 
    # unless specified. The prompt doesn't say "no loops".
    
    # Let's use the DSU logic with while loops.
    
    # We need to store the parents for 4 directions.
    # For row r, col c:
    # Right: parent_R[r][c]
    # Left: parent_L[r][c]
    # Down: parent_D[c][r]
    # Up: parent_U[c][r]
    
    # Using lists of lists (though we must be careful with memory)
    # We initialize them using list comprehensions.
    
    pR = [[c + 1 for c in range(W + 2)] for _ in range(H + 1)]
    pL = [[c - 1 for c in range(W + 2)] for _ in range(H + 1)]
    pD = [[r + 1 for r in range(H + 2)] for _ in range(W + 1)]
    pU = [[r - 1 for r in range(H + 2)] for _ in range(W + 1)]
    
    # Path compression functions
    def find_R(r, c):
        # Use a trick to avoid while loop: recursion is allowed if limit is high
        # But we can also use a while loop.
        curr = c
        path = []
        while curr <= W and pR[r][curr] != curr:
            path.append(curr)
            curr = pR[r][curr]
        for node in path:
            pR[r][node] = curr
        return curr

    def find_L(r, c):
        curr = c
        path = []
        while curr >= 1 and pL[r][curr] != curr:
            path.append(curr)
            curr = pL[r][curr]
        for node in path:
            pL[r][node] = curr
        return curr

    def find_D(c, r):
        curr = r
        path = []
        while curr <= H and pD[c][curr] != curr:
            path.append(curr)
            curr = pD[c][curr]
        for node in path:
            pD[c][node] = curr
        return curr

    def find_U(c, r):
        curr = r
        path = []
        while curr >= 1 and pU[c][curr] != curr:
            path.append(curr)
            curr = pU[c][curr]
        for node in path:
            pU[c][node] = curr
        return curr

    # Process queries
    queries = [list(map(int, input_data[i:i+2])) for i in range(3, len(input_data), 2)]
    
    # We need to track which walls are gone to count them
    # Since we can't use a loop to process queries, we use map()
    
    # State to track total walls
    state = {'count': H * W}
    
    def destroy(r, c):
        # If wall exists, just remove it.
        # To check if wall exists, we can't use a separate array if we can't loop.
        # But we can check if find_R(r, c) == c.
        if find_R(r, c) == c:
            state['count'] -= 1
            # Remove from all 4 DSUs
            # To "remove" c from row r, we make pR[r][c] = find_R(r, c+1)
            # and pL[r][c] = find_L(r, c-1)
            pR[r][c] = find_R(r, c + 1)
            pL[r][c] = find_L(r, c - 1)
            pD