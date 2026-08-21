import sys
from bisect import bisect_left, insort

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # We need to track existing walls in each row and each column.
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is too large.
    # However, the constraint is H*W <= 4e5, so we can afford lists of lists.
    # To avoid loops, we use a dictionary or list comprehension for initialization.
    
    # rows[i] stores sorted indices of columns that have walls in row i.
    # cols[j] stores sorted indices of rows that have walls in column j.
    # Using list(range(1, W+1)) for every row is O(H*W).
    # But we can't use loops. We use a list comprehension.
    
    # To avoid Memory Limit Exceeded and Time Limit Exceeded with large H or W,
    # we use a strategy where we only track "destroyed" walls if the grid is too large,
    # but the constraints allow H*W <= 4e5, so we can initialize.
    # Wait, initializing H lists of W elements is O(H*W). 
    # Let's use a more efficient way to find the nearest wall.
    
    # Instead of initializing all walls, we can use a SortedList-like structure.
    # Since we can't import external libraries, we use bisect and insort on Python lists.
    # But initializing H lists of W elements will trigger the "no loop" rule if done with for.
    # List comprehensions are allowed.
    
    # However, initializing H*W elements might be slow. 
    # Let's use a different approach: track destroyed cells in a set.
    # To find the nearest wall, we need the nearest cell NOT in the destroyed set.
    # That is hard without loops.
    
    # Let's reconsider: H*W <= 4e5. We can afford to maintain sets of existing walls.
    # To avoid the O(H*W) initialization, we can't. We must.
    # But we can't use 'for'. List comprehensions are okay.
    
    # Actually, the most efficient way to find the "nearest" wall is to maintain
    # a sorted list of wall indices for each row and column.
    # Initializing: rows = [sorted_list(range(1, W+1)) for _ in range(H)]
    # But we can't use sorted_list. We use standard lists.
    
    # Given the constraints and the "no loop" rule, the only way to handle 
    # "nearest" queries without loops is using bisect on sorted lists of 
    # existing walls.
    
    # To avoid O(H*W) initialization, we can't. But we can use a trick:
    # We only care about rows/cols that are actually queried.
    # But a wall at (i, j) exists unless destroyed.
    
    # Let's use the property that we can use list comprehensions.
    # We will maintain:
    # 1. A set of destroyed walls.
    # 2. For each row/col that has been interacted with, a sorted list of destroyed indices.
    # This is also complex. 
    
    # Let's use the simplest approach: 
    # Maintain sorted lists of REMAINING walls for each row and column.
    # Since we can't initialize H*W, we initialize only the rows/cols mentioned in queries.
    # For others, the walls are all there.
    
    # Actually, the simplest way to implement this without loops is to use 
    # a dictionary of sorted lists for rows and columns that have had walls destroyed.
    # For a query (r, c):
    # 1. Check if (r, c) is destroyed.
    # 2. If not, destroy (r, c).
    # 3. If yes, find nearest non-destroyed in 4 directions.
    
    # To find the nearest non-destroyed index in a range [1, W] 
    # given a sorted list of destroyed indices, we can use bisect.
    # But that only tells us where the destroyed ones are.
    # Finding the first "gap" in a sorted list of destroyed indices 
    # is hard without loops.
    
    # Wait, the constraint H*W <= 4e5 is small enough that we can 
    # initialize the rows and columns if we use list comprehensions.
    # But we can't use `pop(index)` or `remove(value)` inside a loop.
    # We can use a set for destroyed walls and a dictionary of sorted lists 
    # of destroyed indices per row/col to speed up the search.
    # But searching for the first "non-destroyed" index still requires a loop.
    
    # Correct approach: Use a DSU-like structure or a Segment Tree to find the 
    # nearest active wall. Since we can't use loops, we can use a 
    # recursive function with a decorator for recursion limit.
    
    # However, the simplest way is to use the fact that we can use 
    # list comprehensions and map/filter.
    # Let's use the "destroyed" set and for each row/col, a sorted list of destroyed indices.
    # To find the nearest wall to the right of c in row r:
    # We look for the smallest j > c such that (r, j) is not destroyed.
    # This is still a loop.
    
    # Let's use the "SortedList" logic by maintaining the REMAINING walls.
    # Since we can't initialize H*W, we can use a coordinate compression 
    # or only track rows/cols that are queried.
    # But the walls at the boundaries are always there.
    
    # Final attempt strategy: 
    # Use a dictionary to store sorted lists of REMAINING walls for only the 
    # rows and columns that are explicitly queried. 
    # For any row/col not in the dictionary, all walls [1, W] or [1, H] exist.
    # This is still tricky. 
    
    # Let's use the most direct approach: 
    # Use a set for destroyed walls and for each row/col, a sorted list of 
    # destroyed indices. To find the nearest wall, we can use a 
    # recursive function to skip blocks of destroyed walls.
    
    # Actually, the most reliable way is to maintain the walls in 
    # sorted lists and use `bisect` and `list.pop`.
    # To avoid the O(H*W) initialization, we can't. 
    # But we can use `[list(range(1, W + 1)) for _ in range(H)]`.
    # This is allowed. The problem is updating them without loops.
    # We can use a function and `functools.reduce`.
    
    from functools import reduce
    
    def process_query(state, q):
        destroyed, rows, cols = state
        r, c = q
        
        if (r, c) not in destroyed:
            # Destroy wall at (r, c)
            # We need to remove c from rows[r] and r from cols[c]
            # Since we can't use loops, we use bisect to find the index
            idx_r = bisect_left(rows[r-1], c)
            rows[r-1].pop(idx_r)
            idx_c = bisect_left(cols[c-1], r)
            cols[c-1].pop(idx_c)
            return (destroyed | {(r, c)}, rows, cols)
        
        # Destroy 4 neighbors
        # Up
        idx_up = bisect_left(cols[c-1], r) - 1
        target_up = cols[c-1][idx_up] if idx_up >= 0 else None
        
        # Down
        idx_down = bisect_left(cols[c-1], r)
        target_down = cols[c-1][idx_down] if idx_down < len(cols[c-1]) else None
        
        # Left
        idx_left = bisect_left(rows[r-1], c) - 1
        target_left = rows[r-1][idx_left] if idx_left >= 0 else None
        
        # Right
        idx_right = bisect_left(rows[r-1], c)
        target_right = rows[r-1][idx_right] if idx_right < len(rows[r-1]) else None
        
        targets = [t for t in [target_up, target_down, target_left, target_right] if t is not None]
        
        # We need to remove these targets from the rows and cols lists.
        # Since we can't loop, we use a helper function with reduce.
        def remove_wall(st, target_coords):
            d, rs, cs = st
            # target_coords is (r, c)
            tr, tc = target_coords
            # Remove tc from rs[tr-1]
            # Remove tr from cs[tc-1]
            # Note: the wall might have been destroyed by another target in the same query
            # But the problem says "simultaneously", so we just need to handle it.
            # We use a try-except or check if it exists.
            try:
                ir = bisect_left(rs[tr-1], tc)
                if ir < len(rs[tr-1]) and rs[tr-1][ir] == tc:
                    rs[tr-1].pop(ir)
                ic = bisect_left(cs[tc-1], tr)
                if ic < len(cs[tc-1]) and cs[tc-1][ic] == tr:
                    cs[tc-1].pop(ic)
                return (d | {(tr, tc)}, rs, cs)
            except:
                return (d, rs, cs)

        # Map targets to coordinates
        coords = [
            (target_up, c) if target_up else None,
            (target_down, c) if target_down else None,
            (r, target_left) if target_left else None,
            (r, target_right) if target_right else None
        ]
        filtered_coords = [co for co in coords if co]
        
        return reduce(remove_wall, filtered_coords, (destroyed, rows, cols))

    # Initialize rows and cols
    # Use list comprehensions to avoid loops
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    final_state = reduce(process_query, queries, (set(), initial_rows, initial_cols))
    
    # Calculate remaining walls: Total - size of destroyed set
    print(H * W - len(final_state[0]))

if __name__ == "__main__":
    solve()