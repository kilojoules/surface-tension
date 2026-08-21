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
    # Using sorted lists to allow binary search for the nearest wall.
    # rows[i] contains indices of columns that have walls in row i.
    # cols[j] contains indices of rows that have walls in column j.
    rows = [sorted(range(1, W + 1)) for _ in range(H + 1)]
    cols = [sorted(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions from sorted lists efficiently without using loops,
    # we use a technique where we don't physically delete from the list 
    # (which is O(N)) but rather use a data structure or a different approach.
    # However, given the constraints (H*W <= 4e5) and the specific "nearest" requirement,
    # we can use a Doubly Linked List approach simulated with arrays to get O(1) deletion.
    
    # For each cell (i, j), we store the index of the next/prev wall in that row/col.
    # L[i][j], R[i][j], U[i][j], D[i][j]
    # To save memory (H*W is 4e5), we use flat arrays.
    # Indexing: (i-1)*W + (j-1)
    
    # Since we cannot use loops or recursion, we use map/list comprehensions.
    # We initialize the "pointers" to the neighbors.
    # L: left neighbor column, R: right, U: up, D: down.
    
    # Using a dictionary or a flat list to track if a wall exists.
    # wall_exists = [True] * (H * W)
    
    # Given the constraints and the "no loop" requirement, the most effective way 
    # to find the "nearest" wall without loops is using a data structure that 
    # supports fast updates and queries. 
    # Since we can't use loops, we can use a mutable state and a functional 
    # approach via map/list comprehensions or a helper class with a mutable state.
    
    # Let's use a class to maintain the state and a list comprehension to iterate through queries.
    
    class GridState:
        def __init__(self, h, w):
            self.h = h
            self.w = w
            # We use dictionaries to store the "links" for each row and column.
            # row_links[r] = {col: [left, right]}
            # col_links[c] = {row: [up, down]}
            # To avoid loops, we initialize these using comprehensions.
            self.row_links = {r: {c: [c-1, c+1] for c in range(1, w+1)} for r in range(1, h+1)}
            self.col_links = {c: {r: [r-1, r+1] for r in range(1, h+1)} for c in range(1, h+1)}
            self.exists = {(r, c): True for r in range(1, h+1) for c in range(1, w+1)}
            self.remaining = h * w

        def remove_wall(self, r, c):
            if not (1 <= r <= self.h and 1 <= c <= self.w) or not self.exists[(r, c)]:
                return
            
            # Get neighbors
            left, right = self.row_links[r][c]
            up, down = self.col_links[c][r]
            
            # Update neighbors to point to each other
            if left >= 1: self.row_links[r][left][1] = right
            if right <= self.w: self.row_links[r][right][0] = left
            if up >= 1: self.col_links[c][up][1] = down
            if down <= self.h: self.col_links[c][down][0] = up
            
            self.exists[(r, c)] = False
            self.remaining -= 1

        def process_query(self, q):
            r, c = q
            if self.exists[(r, c)]:
                self.remove_wall(r, c)
            else:
                # Find nearest walls in 4 directions
                # Since the wall at (r, c) is already gone, we need to find the 
                # closest existing walls relative to (r, c).
                # We can't loop, but we can check the links of the "empty" cell.
                # Wait, the links must be maintained even for empty cells to find the next wall.
                # Let's redefine: row_links[r][c] always points to the nearest existing walls.
                
                # To find the nearest wall to the left of (r, c):
                # We need to know which wall was to the left of (r, c) when it was destroyed.
                # Let's use a different approach: 
                # For each row, a sorted list of existing walls. 
                # Since we can't loop, we use bisect and list.pop().
                # But list.pop(i) is O(N). 
                # However, we can use a Fenwick tree or Segment tree? No, those are for sums.
                # A balanced BST? Python doesn't have one built-in.
                # Let's use the property that we can use `set` and `bisect` if we convert to list.
                # But converting to list is O(N).
                
                # Correct approach for "no loops": 
                # Use a mutable object and `map` or list comprehensions.
                # To handle the "nearest" wall, we can use a dictionary to simulate 
                # a doubly linked list for every row and column.
                pass

    # Redesigning to fit "no loops" and "no recursion" strictly:
    # We will use a class to hold the state and a list comprehension to drive the queries.
    
    class FastGrid:
        def __init__(self, h, w):
            self.h, self.w = h, w
            # Store only existing walls in each row/col using sets
            # To find the nearest, we can't use sets. We need sorted lists.
            # Since we can't use loops, we'll use a dictionary to simulate the 
            # Doubly Linked List for every single row and column.
            self.row_prev = {r: {c: c - 1 for c in range(1, w + 1)} for r in range(1, h + 1)}
            self.row_next = {r: {c: c + 1 for c in range(1, w + 1)} for r in range(1, h + 1)}
            self.col_prev = {c: {r: r - 1 for r in range(1, h + 1)} for c in range(1, w + 1)}
            self.col_next = {c: {r: r + 1 for r in range(1, h + 1)} for c in range(1, w + 1)}
            self.exists = {(r, c): True for r in range(1, h + 1) for c in range(1, w + 1)}
            self.count = h * w

        def delete(self, r, c):
            if not (1 <= r <= self.h and 1 <= c <= self.w) or not self.exists[(r, c)]:
                return
            
            lp, rp = self.row_prev[r][c], self.row_next[r][c]
            up, dp = self.col_prev[c][r], self.col_next[c][r]
            
            if rp <= self.w: self.row_prev[r][rp] = lp
            if lp >= 1: self.row_next[r][lp] = rp
            if dp <= self.h: self.col_prev[c][dp] = up
            if up >= 1: self.col_next[c][up] = dp
            
            self.exists[(r, c)] = False
            self.count -= 1

        def query(self, q):
            r, c = q
            if self.exists[(r, c)]:
                self.delete(r, c)
            else:
                # Find nearest walls using the links of the cell (r, c)
                # Note: (r, c) was already deleted, so its links still point to 
                # the walls that were its neighbors at the time of its deletion.
                # But those neighbors might have been deleted too.
                # To fix this, we need a way to find the CURRENT nearest wall.
                # The DLL approach only works if we delete the current node.
                # If we want to find the nearest wall to an empty cell, 
                # we need to traverse. But loops are forbidden.
                # Wait, we can use a recursive-like structure with map, but recursion is forbidden.
                # Let's use the property: we can use a while loop? No, "loops are forbidden".
                # "For" and "While" are forbidden.
                pass

# The constraint "no loops" is extremely strict. 
# However, we can use `bisect` on sorted lists. 
# To avoid O(N) deletion, we can use a `SortedList` from `sortedcontainers`, 
# but that's not standard library.
# Standard library alternative: `bisect` with a list, and instead of deleting, 
# we can't. But we can use a Fenwick tree to find the k-th element? No.
# Let's use the `bisect` module and accept that we can't use loops.
# We can use `list.pop()` which is O(N), but we can't use loops to call it.
# Actually, the only way to handle this without loops/recursion is to use 
# map/filter/reduce and a mutable state.
# But the "nearest wall" problem is fundamentally sequential or requires a 
# data structure that supports O(log N) deletion and successor/predecessor queries.
# Python's `bisect` works on lists. We can maintain sorted lists of walls.
# To delete without a loop, we can use `del list[index]`.

import sys
from bisect import bisect_left, bisect_right

def main():
    input_data = sys.stdin.read().split()
    if not input_data: return
    H, W, Q = map(int, input_data[:3])
    queries = list(map(int, input_data[3:]))
    
    # Group queries into pairs
    query_pairs = list(zip(queries[0::2], queries[1::2]))
    
    # State
    rows = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    def remove_wall(r, c):
        # Use bisect to find index and del to remove
        # This is O(W) or O(H), but we can't use loops.
        # We must use a trick to call this.
        idx_r = bisect_left(rows[r], c)
        if idx_r < len(rows[r]) and rows[r][idx_r] == c:
            del rows[r][idx_r]
        idx_c = bisect_left(cols[c], r)
        if idx_c < len(cols[c]) and cols[c][idx_c] == r:
            del cols[c][idx_c]

    def handle_query(q):
        r, c = q
        # Check if wall exists
        idx_r = bisect_left(rows[r], c)
        exists = (idx_r < len(rows[r]) and rows[r][idx_r] == c)
        
        if exists:
            remove_wall(r, c)
        else:
            # Find neighbors
            # Left: element at idx_r - 1
            # Right: element at idx_r
            # Up: element at idx_c - 1
            # Down: element at idx_c
            idx_c = bisect_left(cols[c], r)
            
            # We need to collect all walls to be deleted first to avoid 
            # index shifting during deletion.
            targets = [
                (r, rows[r][idx_r - 1]) if idx_r > 0 else None,
                (r