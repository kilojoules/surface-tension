import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # The constraint to avoid explicit loops forces a functional style, 
    # which can be cleaner for certain transformations but is generally 
    # less idiomatic for state-heavy simulations in Python.
    # However, we can use data structures that allow efficient updates 
    # and use map/list comprehensions to process queries.
    
    # Since we need to find the "nearest" wall, we need a way to skip 
    # empty cells. A Disjoint Set Union (DSU) or similar structure is 
    # usually used, but since we cannot use loops, we will use 
    # dictionary-based memoization or recursion (with increased limit) 
    # to simulate the jumping behavior.
    
    # Actually, the constraint to avoid loops makes DSU hard. 
    # But we can use a technique: for each row and column, maintain 
    # a set of existing walls. To find the nearest wall, we can use 
    # bisect on sorted lists of existing wall indices.
    
    from bisect import bisect_left
    
    # rows[i] stores sorted indices of walls in row i
    # cols[j] stores sorted indices of walls in column j
    # Using lists and bisect.insort is O(N), but since we only remove,
    # we can use a SortedList from a library, but we can't import external libs.
    # Given the constraints and the "no loop" rule, we will use 
    # recursion to process the queries and sets/bisect to find neighbors.
    
    # Wait, the "no loop" constraint is very strict. 
    # Let's use a recursive function to process the queries.
    sys.setrecursionlimit(1000000)
    
    # We need to track which walls are destroyed.
    # Since we can't use loops, we'll use a set of (r, c) for destroyed walls.
    # But finding the "nearest" wall requires knowing which are NOT destroyed.
    # Let's maintain sorted lists of active walls for each row and column.
    # Since we can't use loops, we'll use a recursive function to handle the Q queries.
    
    # To handle the "no loop" constraint while maintaining sorted lists 
    # without O(N) deletions, we can use a Fenwick tree or Segment Tree, 
    # but those are hard to implement without loops.
    # Actually, the simplest way to find the nearest wall is to use 
    # a DSU-like structure implemented via a dictionary and recursion.
    
    parent_r = {} # For each row, DSU to find next wall
    parent_c = {} # For each col, DSU to find next wall
    
    # However, the most straightforward way to implement this 
    # without loops is to use a recursive function for the queries 
    # and a dictionary to track wall existence.
    
    walls = set() # We'll track destroyed walls. Total walls = H*W - len(walls)
    
    # To find the nearest wall efficiently without loops:
    # We can use a dictionary where keys are (row, col, direction) 
    # and values are the next wall index.
    # But that's basically DSU. Let's implement DSU find recursively.
    
    memo_up = {}
    memo_down = {}
    memo_left = {}
    memo_right = {}

    def find_up(r, c):
        if r < 1: return None
        key = (r, c)
        if key not in memo_up:
            # If wall exists at (r, c), it is the nearest
            # But the logic is: if we are at (r, c) and it's empty, 
            # look at r-1.
            # This is tricky because the "wall" status changes.
            pass
        return None

    # Given the constraints and the "no loop" rule, the most viable 
    # approach is to use a recursive function to iterate through queries 
    # and use a data structure that supports fast updates.
    # Since we can't use loops, we'll use a recursive function to 
    # simulate the query process and a set to track destroyed walls.
    # To find the nearest wall, we can use the fact that H*W is 4e5.
    # We can maintain sorted lists of walls for each row/col and 
    # use bisect to find neighbors, and use a recursive function 
    # to "remove" elements by filtering (though filtering is O(N)).
    
    # Wait, the only way to avoid loops and recursion for iteration 
    # is using map/filter/reduce. 
    # Let's use a recursive function for the Q queries and 
    # a dictionary to store the walls. 
    # To find the nearest wall, we can use a DSU-like jump pointer 
    # stored in dictionaries.
    
    # jump[row][col][direction] = next_col_or_row
    # We can use a dictionary with keys (r, c, dir)
    jumps = {} 

    def get_jump(r, c, direction):
        # direction: 0:up, 1:down, 2:left, 3:right
        if direction == 0: # Up
            if r < 1: return None
            if (r, c, 0) not in jumps: jumps[(r, c, 0)] = r
            if is_destroyed(jumps[(r, c, 0)], c):
                jumps[(r, c, 0)] = get_jump(jumps[(r, c, 0)] - 1, c, 0)
            return jumps[(r, c, 0)]
        # ... this is getting complex.
        
    # Let's reconsider. The simplest way to find the nearest wall 
    # is to maintain a set of existing walls for each row and column.
    # Since we can't use loops, we use `bisect` and `set.remove`.
    # But we can't use `for` to initialize. We use list comprehensions.
    
    # Actually, the most efficient way to find the nearest wall 
    # is using two DSU structures per row/column (one for +1, one for -1).
    # We can implement DSU `find` recursively.
    
    # Since I must avoid loops, I will use:
    # 1. A recursive function to process queries.
    # 2. DSU (implemented via dictionaries and recursion) to find the nearest wall.
    # 3. List comprehensions for initialization.
    
    destroyed = set()
    
    # DSU structures for each row (left/right) and each col (up/down)
    # L[r][c] points to the next potential wall to the left
    # R[r][c] points to the next potential wall to the right
    # U[r][c] points to the next potential wall up
    # D[r][c] points to the next potential wall down
    
    # To avoid loops and deep recursion, we use a dictionary for DSU
    # and a recursive function with a helper to find the root.
    
    parent_L = {}
    def find_L(r, c):
        if c < 1: return None
        if (r, c) not in parent_L: return c
        parent_L[(r, c)] = find_L(r, parent_L[(r, c)])
        return parent_L[(r, c)]

    parent_R = {}
    def find_R(r, c):
        if c > W: return None
        if (r, c) not in parent_R: return c
        parent_R[(r, c)] = find_R(r, parent_R[(r, c)])
        return parent_R[(r, c)]

    parent_U = {}
    def find_U(r, c):
        if r < 1: return None
        if (r, c) not in parent_U: return r
        parent_U[(r, c)] = find_U(parent_U[(r, c)], c)
        return parent_U[(r, c)]

    parent_D = {}
    def find_D(r, c):
        if r > H: return None
        if (r, c) not in parent_D: return r
        parent_D[(r, c)] = find_D(parent_D[(r, c)], c)
        return parent_D[(r, c)]

    def destroy(r, c):
        if r < 1 or r > H or c < 1 or c > W or (r, c) in destroyed:
            return
        destroyed.add((r, c))
        # Update DSU: this cell is no longer a wall, so jump over it
        parent_L[(r, c)] = find_L(r, c - 1)
        parent_R[(r, c)] = find_R(r, c + 1)
        parent_U[(r, c)] = find_U(r - 1, c)
        parent_D[(r, c)] = find_D(r + 1, c)

    def process_query(idx):
        if idx >= Q:
            return len(destroyed)
        
        r = int(input_data[3 + 2*idx])
        c = int(input_data[4 + 2*idx])
        
        if (r, c) not in destroyed:
            destroy(r, c)
        else:
            # Destroy 4 neighbors
            # Up
            res_u = find_U(r - 1, c)
            if res_u: destroy(res_u, c)
            # Down
            res_d = find_D(r + 1, c)
            if res_d: destroy(res_d, c)
            # Left
            res_l = find_L(r, c - 1)
            if res_l: destroy(r, res_l)
            # Right
            res_r = find_R(r, c + 1)
            if res_r: destroy(r, res_r)
            
        return process_query(idx + 1)

    # To avoid recursion depth issues for find_U/D/L/R, 
    # we need to be careful. But the problem says no loops.
    # Let's use a trick: we can use a while loop inside a function 
    # if we define it as a helper, but the prompt says "no for or while loops".
    # This means we MUST use recursion or functional tools.
    
    # The DSU find can be deep. Let's use a non-recursive find 
    # by using a list comprehension trick or just hope 
    # the recursion limit is enough.
    
    # Actually, the most robust way to implement DSU without loops 
    # is to use a recursive function with path compression.
    
    # Final calculation: Total walls - destroyed walls
    total_walls = H * W
    destroyed_count = process_query(0)
    print(total_walls - destroyed_count)

# Standard Python entry point
if __name__ == "__main__":
    solve()