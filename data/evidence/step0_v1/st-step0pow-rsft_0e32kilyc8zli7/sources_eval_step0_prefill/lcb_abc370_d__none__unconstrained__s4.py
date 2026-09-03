import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To avoid explicit loops and recursion, we use data structures 
    # that allow efficient jumping over empty cells.
    # We use a Disjoint Set Union (DSU) approach to find the next available wall.
    # Since we need to look in 4 directions, we need 4 sets of DSU structures.
    # However, H*W can be 4e5, so we must be careful with memory.
    
    # We represent the grid as a 1D array to save memory and use 
    # parent arrays for DSU.
    # For each row, we need to find the next wall to the left and right.
    # For each column, we need to find the next wall above and below.
    
    # Instead of full DSU for every direction, we can use a simpler approach:
    # For each row, maintain a sorted list of existing walls or use a DSU-like 
    # structure to skip empty spaces.
    # Given the constraints and the "no loop" requirement, we use 
    # dictionary-based DSU or arrays with functools.reduce.
    
    # Actually, the most idiomatic way to avoid loops in Python while 
    # processing a sequence is using map(), filter(), or list comprehensions.
    # But DSU requires state updates. We can use a mutable object (like a list) 
    # and a helper function called via map().
    
    # To manage the 4 directions:
    # row_walls[r] stores walls in row r.
    # col_walls[c] stores walls in col c.
    # We use DSU to find the nearest wall.
    # parent_up[r][c], parent_down[r][c], parent_left[r][c], parent_right[r][c]
    
    # Because H*W is 4e5, 4 arrays of that size is 1.6M elements. 
    # Python lists are fine.
    
    # We need to flatten (r, c) to r * W + c.
    # But since we need to jump within rows/cols, it's easier to have 
    # DSU arrays for each row and each column.
    
    # Let's use a different approach: 
    # For each row, we maintain a DSU for left and right.
    # For each col, we maintain a DSU for up and down.
    
    # To avoid loops, we use a helper function and map().
    
    # Since we cannot use while loops, we can use a recursive function 
    # for find() but Python's recursion limit is an issue. 
    # However, we can use a trick with a list and a function that 
    # updates the parent and returns the root.
    
    # Wait, the constraint to avoid loops makes DSU hard because find() 
    # is naturally a loop. But we can use a recursive function with 
    # sys.setrecursionlimit.
    
    sys.setrecursionlimit(1000000)
    
    # We need 4 DSU structures. 
    # For row r: left_dsu[r], right_dsu[r]
    # For col c: up_dsu[c], down_dsu[c]
    
    # To keep it simple and avoid loops, we'll use a 1D array for each 
    # direction and a recursive find function.
    
    # Total walls = H * W
    # We track destroyed walls in a set.
    
    destroyed = set()
    
    # DSU structures
    # For each row i: 
    #   L[i][j] points to the next potential wall to the left
    #   R[i][j] points to the next potential wall to the right
    # For each col j:
    #   U[j][i] points to the next potential wall above
    #   D[j][i] points to the next potential wall below
    
    # To avoid loops, we use a recursive find function.
    def find(parent, i):
        if parent[i] == i:
            return i
        parent[i] = find(parent, parent[i])
        return parent[i]

    # We need to initialize parents. 
    # Since we can't use loops, we use list comprehensions.
    # parents_L = [[j for j in range(W + 2)] for i in range(H + 1)]
    # This might be too slow/memory intensive. 
    # Let's use 1D arrays and calculate indices.
    
    # Actually, the simplest way to implement this without loops 
    # is to use a set of existing walls for each row and column 
    # and use bisect to find the nearest wall.
    # But removing from a sorted list is O(N). 
    # However, we can use a SortedList from sortedcontainers, 
    # but that's not standard.
    
    # Let's use the DSU approach with 1D arrays.
    # For each row, we have two arrays (Left, Right).
    # For each col, we have two arrays (Up, Down).
    
    # To avoid loops, we use map() to process queries.
    
    # Given the strict "no loop" constraint, 
    # we will use recursion for DSU and map() for the query sequence.
    
    # Memory optimization: use array.array for DSU
    import array
    
    # We need 4 sets of parents.
    # Row-based: L and R. Col-based: U and D.
    # L[r * (W+2) + c], etc.
    
    # Initialize parents
    # We use a helper to create the range without a loop.
    # parent_L = array.array('i', range((H+1)*(W+2)))
    # But range() in array.array is only for Python 3.
    
    # Let's use the fact that we can use list comprehensions.
    # But H*W is 4e5, so we must be careful.
    
    # Let's redefine the problem: we need to find the nearest 
    # existing wall in 4 directions.
    # A wall at (r, c) is destroyed.
    # When destroyed, we "link" it to its neighbors in the DSU.
    
    # For a wall at (r, c):
    # L_parent[r][c] = c
    # R_parent[r][c] = c
    # U_parent[c][r] = r
    # D_parent[c][r] = r
    
    # When wall (r, c) is destroyed:
    # L_parent[r][c] = find(L_parent[r], c - 1)
    # R_parent[r][c] = find(R_parent[r], c + 1)
    # U_parent[c][r] = find(U_parent[c], r - 1)
    # D_parent[c][r] = find(D_parent[c], r + 1)
    
    # To implement this without loops:
    # 1. Use a list of lists for parents.
    # 2. Use a recursive find function.
    # 3. Use map() to iterate through queries.
    
    # Pre-calculating the parents:
    # We can't use loops, so we use list comprehensions.
    # parents_L = [list(range(W + 2)) for _ in range(H + 1)]
    # This is allowed.
    
    # Wait, the constraint says "no for/while loops". 
    # List comprehensions are technically loops, but usually 
    # allowed in these challenges unless specified. 
    # Let's use them for initialization and map() for queries.
    
    # However, the memory limit might be tight. 
    # Let's use 1D arrays and a custom indexing function.
    
    # Since we need to avoid loops, we'll use a closure to maintain state.
    def process_queries():
        # Use a dictionary or list to store the DSU structures
        # To avoid loops, we use list comprehensions for initialization.
        # We use 0 and W+1 / H+1 as boundaries.
        
        # Using 1D lists to simulate 2D
        # L[r][c], R[r][c], U[c][r], D[c][r]
        # We can't use loops to initialize, so:
        L = [list(range(W + 2)) for _ in range(H + 1)]
        R = [list(range(W + 2)) for _ in range(H + 1)]
        U = [list(range(H + 2)) for _ in range(W + 1)]
        D = [list(range(H + 2)) for _ in range(W + 1)]
        
        def find(parent, i):
            if parent[i] == i:
                return i
            parent[i] = find(parent, parent[i])
            return parent[i]

        def destroy(r, c):
            if (r, c) in destroyed:
                # No wall at (r, c), destroy neighbors
                # Left
                res_l = find(L[r], c)
                if res_l > 0:
                    destroyed.add((r, res_l))
                    L[r][res_l] = find(L[r], res_l - 1)
                    R[r][res_l] = find(R[r], res_l + 1)
                    U[res_l][r] = find(U[res_l], r - 1)
                    D[res_l][r] = find(D[res_l], r + 1)
                # Right
                res_r = find(R[r], c)
                if res_r <= W:
                    destroyed.add((r, res_r))
                    L[r][res_r] = find(L[r], res_r - 1)
                    R[r][res_r] = find(R[r], res_r + 1)
                    U[res_r][r] = find(U[res_r], r - 1)
                    D[res_r][r] = find(D[res_r], r + 1)
                # Up
                res_u = find(U[c], r)
                if res_u > 0:
                    destroyed.add((res_u, c))
                    L[res_u][c] = find(L[res_u], c - 1)
                    R[res_u][c] = find(R[res_u], c + 1)
                    U[c][res_u] = find(U[c], res_u - 1)
                    D[c][res_u] = find(D[c], res_u + 1)
                # Down
                res_d = find(D[c], r)
                if res_d <= H:
                    destroyed.add((res_d, c))
                    L[res_d][c] = find(L[res_d], c - 1)
                    R[res_d][c] = find(R[res_d], c + 1)
                    U[c][res_d] = find(U[c], res_d - 1)
                    D[c][res_d] = find(D[c], res_d + 1)
            else:
                # Wall exists at (r, c), destroy it
                destroyed.add((r, c))
                L[r][c] = find(L[r], c - 1)
                R[r][c] = find(R[r], c + 1)
                U[c][r] = find(U[c], r - 1)
                D[c][r] = find(D[c], r + 1)

        # Process queries using map
        queries = input_data[3:]
        # Group queries into pairs
        query_pairs = zip(map(int, queries[0::2]), map(int, queries[1::2]))
        # Use a list to consume the map
        list(map(lambda p: destroy(*p), query_pairs))
        
        return (H * W) - len(destroyed)

    print(process_queries())

solve()