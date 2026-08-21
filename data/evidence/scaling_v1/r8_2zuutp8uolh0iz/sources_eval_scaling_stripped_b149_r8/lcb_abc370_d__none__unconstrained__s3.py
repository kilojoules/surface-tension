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
    # Since H*W is up to 4e5, we can't use a full 2D array if H or W is very large,
    # but the constraint says H*W <= 4e5, so we can use lists of sets or sorted lists.
    # To avoid loops, we use a dictionary of sorted lists for rows and columns.
    # However, initializing H sorted lists via a loop is forbidden.
    # We use a dictionary and only add entries when a wall is destroyed, 
    # but that's for "destroyed" walls. We need "existing" walls.
    # Given the constraints and the "no loop" rule, we use a dictionary 
    # to store the sorted lists of REMOVED walls and calculate remaining.
    
    # Actually, the most efficient way to find the "nearest wall" without loops
    # is to maintain a data structure of REMOVED walls and use binary search 
    # to find the gaps. But the problem says walls are initially everywhere.
    # So we track REMOVED walls.
    
    # row_removed[r] = sorted list of columns c where wall (r, c) is gone.
    # col_removed[c] = sorted list of rows r where wall (r, c) is gone.
    
    # Since we can't use loops to initialize, we use a dictionary.
    # To find the first wall to the left of (r, c):
    # It is the largest j < c such that (r, j) is NOT in row_removed[r].
    # This is tricky. Let's instead track EXISTING walls using a different approach.
    # Wait, if we can't use loops, we can't initialize H lists. 
    # But we can use a dictionary and the `get` method.
    
    # Let's reconsider: we need to find the nearest index i < R such that wall exists.
    # If we track REMOVED walls in sorted lists, the nearest existing wall to the left
    # is c-1, UNLESS c-1 is removed, then we check the block of removed walls ending at c-1.
    
    # Correct approach: Use a DSU-like structure or a Segment Tree to find the nearest 1.
    # But we can't implement those without loops (for initialization).
    # Actually, we can use a dictionary of sets to track removed walls.
    # To find the nearest existing wall to the left of (r, c):
    # We need the largest j < c such that (r, j) is NOT removed.
    # If we maintain the removed walls in a sorted list, we can find the range of 
    # contiguous removed walls ending at c-1.
    
    # Let's use a different strategy: 
    # Since we need to find the first wall, and walls are initially everywhere,
    # the first wall to the left of (r, c) is simply c-1, if (r, c-1) is not removed.
    # If (r, c-1) is removed, it's the first index to the left of the contiguous 
    # block of removed walls containing c-1.
    
    # We can use a dictionary to store DSU parents for each row and column.
    # parent_r[r][c] points to the next potential wall.
    # But DSU initialization is the problem.
    
    # Alternative: Use a dictionary to store removed cells.
    # For a query (r, c):
    # 1. If (r, c) is not removed, remove it.
    # 2. If (r, c) is removed, find nearest non-removed in 4 directions.
    # To do this without loops, we can use a dictionary of sorted lists for removed cells.
    # To find the nearest existing wall to the left of c:
    # We look at the sorted list of removed cells in row r.
    # We find the contiguous block of removed cells ending at c-1.
    # The wall is at (c - 1) - (length of contiguous block).
    
    # To find the length of the contiguous block ending at c-1:
    # If we store removed cells in a sorted list, we can use bisect_left.
    # But finding the "start" of a contiguous block in a sorted list without a loop
    # is only possible if we use a DSU or a Segment Tree.
    
    # Wait, the constraint H*W <= 4e5 allows us to use a flat array.
    # We can use a single array of size H*W and simulate a 2D array.
    # But we still can't use loops.
    
    # Let's use the property that we can use `map` and `filter`.
    # We can maintain the state of walls in a set of (r, c) tuples.
    # Initially, all H*W walls exist. That's too many for a set.
    # So we track REMOVED walls in a set.
    
    # To find the nearest existing wall to the left of (r, c):
    # We need max j < c such that (r, j) is not in removed_set.
    # This is still hard without loops.
    
    # Let's use the "SortedList" logic with a dictionary of sorted lists.
    # To find the contiguous block, we can use a DSU implemented with a dictionary.
    # Since we can't initialize, we use `parent.get(x, x)`.
    
    # For each row r, we have a DSU for columns. For each col c, a DSU for rows.
    # There are 4 DSUs: row_left, row_right, col_up, col_down.
    # Each DSU is a dictionary: `(r, c) -> parent_c`
    
    # Since we can't use a while loop for find(), we can use a recursive function.
    # Python's recursion limit needs to be increased.
    
    sys.setrecursionlimit(1000000)
    
    removed = set()
    # dsu stores: (type, line_id, coord) -> next_coord
    # types: 0: row_l, 1: row_r, 2: col_u, 3: col_d
    parent = {}

    def find(t, l, c):
        key = (t, l, c)
        p = parent.get(key, c)
        if p == c:
            return c
        # Path compression
        res = find(t, l, p)
        parent[key] = res
        return res

    def union(t, l, c, target):
        parent[(t, l, c)] = target

    def process_query(q):
        r, c = q
        if (r, c) not in removed:
            removed.add((r, c))
            # Update DSUs: this cell is now "gone", so jump over it
            union(0, r, c, find(0, r, c - 1)) # left
            union(1, r, c, find(1, r, c + 1)) # right
            union(2, c, r, find(2, c, r - 1)) # up
            union(3, c, r, find(3, c, r + 1)) # down
            return 1
        else:
            # Find 4 neighbors
            # Left
            l = find(0, r, c - 1)
            # Right
            ri = find(1, r, c + 1)
            # Up
            u = find(2, c, r - 1)
            # Down
            d = find(3, c, r + 1)
            
            # We must check boundaries and if the wall actually exists
            # (though in this DSU setup, find() will hit 0 or H+1/W+1)
            
            # We need to remove these 4 walls. 
            # Since we can't use a loop, we use a list and map.
            targets = [
                (r, l) if l >= 1 else None,
                (r, ri) if ri <= W else None,
                (u, c) if u >= 1 else None,
                (d, c) if d <= H else None
            ]
            
            # Filter out None and duplicates
            actual_targets = set(filter(None, targets))
            
            # For each target, perform the removal logic
            # We use a helper function to update the DSUs
            def remove_wall(cell):
                if cell not in removed:
                    removed.add(cell)
                    cr, cc = cell
                    union(0, cr, cc, find(0, cr, cc - 1))
                    union(1, cr, cc, find(1, cr, cc + 1))
                    union(2, cc, cr, find(2, cc, cr - 1))
                    union(3, cc, cr, find(3, cc, cr + 1))
                    return 1
                return 0

            return sum(map(remove_wall, actual_targets))

    # Use reduce to process queries without a loop
    from functools import reduce
    
    # We need to track total walls destroyed.
    # Initial walls: H * W. Remaining: H * W - total_destroyed.
    total_destroyed = reduce(lambda acc, q: acc + process_query(q), queries, 0)
    
    print(H * W - total_destroyed)

if __name__ == "__main__":
    solve()