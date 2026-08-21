```python
import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries are pairs of (R, C) starting from index 3
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # We need to track existing walls in every row and every column.
    # Since H*W <= 4*10^5, we can maintain sorted lists of wall indices.
    # rows_walls[r] contains sorted column indices of walls in row r.
    # cols_walls[c] contains sorted row indices of walls in column c.
    # Using 0-indexing internally: row 0 to H-1, col 0 to W-1.
    
    # To efficiently remove elements from sorted lists, we use a set for 
    # existence checks and a sorted list for range queries.
    # However, Python's list.pop(i) is O(N). Given the constraints, 
    # we need a way to find and remove elements efficiently.
    # Since we cannot use external libraries like SortedList, 
    # and H*W is relatively small, we can use sets and 
    # maintain the sorted order via bisect on a list.
    
    # Initialize walls: every cell has a wall.
    # To avoid O(H*W) initialization of lists, we use a different approach.
    # We only care about walls that are actually destroyed.
    # But the problem asks for remaining walls.
    # Total walls = H * W. Remaining = Total - Destroyed.
    
    # We use sets to track which walls are destroyed.
    # destroyed_set = {(r, c), ...}
    # To find the "nearest" wall, we need the sorted indices of REMAINING walls.
    # Since we start with ALL walls, the initial state is range(1, W+1) for every row.
    # Instead of lists, we can use a Fenwick tree or Segment tree to find the 
    # nearest 1 in a bit-array, but that's complex.
    # Given the constraints and Python, the most efficient way to handle 
    # "find nearest" in a dense grid is using a data structure that supports 
    # efficient deletion and successor/predecessor queries.
    
    # Since we can't use SortedList, we can use a dictionary of sets 
    # and maintain sorted lists of remaining walls. 
    # Wait, if we start with all walls, the sorted lists are just range(1, W+1).
    # Removing an element from a Python list is O(N). With N=4*10^5, 
    # Q=2*10^5, O(Q*N) is too slow.
    
    # Correct approach for Python without SortedList:
    # Use a set for existence and a dictionary of sorted lists.
    # To keep deletions efficient, we can't use list.remove().
    # However, we can use the `bisect` module on sorted lists.
    # To handle deletions in O(1) or O(log N), we need a different structure.
    # But we can use a "lazy" deletion approach or a Square Root Decomposition.
    
    # Actually, the most Pythonic way to handle this without external libs 
    # is to use a set for existence and maintain the sorted lists, 
    # but only update them periodically or use a different logic.
    # Let's use a different approach: 
    # For each row and column, maintain a sorted list of existing wall indices.
    # Since we start with all walls, we can't initialize lists of size W.
    # Wait, H*W <= 4*10^5. We CAN initialize lists. 
    # The bottleneck is `list.pop(i)`. 
    # In Python, `list.pop(i)` for a list of size 4*10^5 is actually quite fast 
    # if the total number of pops is 2*10^5, but it's still O(N) worst case.
    # Let's try it; Python's list operations are highly optimized.
    
    # Using 1-based indexing as provided in input
    row_walls = [sorted(range(1, W + 1)) for _ in range(H)]
    col_walls = [sorted(range(1, H + 1)) for _ in range(W)]
    
    # To check if a wall exists in O(1), we use a set of (r, c)
    # But we start with ALL walls, so we track DESTROYED walls.
    destroyed = set()
    
    # Function to remove a wall
    def remove_wall(r, c):
        destroyed.add((r, c))
        # Find index using bisect and remove
        # r and c are 1-indexed.
        # row_walls is 0-indexed list of lists.
        idx_c = bisect_left(row_walls[r-1], c)
        if idx_c < len(row_walls[r-1]) and row_walls[r-1][idx_c] == c:
            row_walls[r-1].pop(idx_c)
            
        idx_r = bisect_left(col_walls[c-1], r)
        if idx_r < len(col_walls[c-1]) and col_walls[c-1][idx_r] == r:
            col_walls[c-1].pop(idx_r)

    for r, c in queries:
        if (r, c) not in destroyed:
            # Check if wall exists at (r, c)
            # Since we track destroyed, if (r, c) not in destroyed, wall exists.
            # Wait, the logic is: if wall exists, destroy it.
            # If it doesn't, destroy 4 neighbors.
            
            # We need to check if the wall at (r, c) is currently there.
            # We can use a helper to check existence.
            # Since we use a set of destroyed walls:
            # Wall exists if (r, c) NOT in destroyed.
            
            # But the problem says: "If there is a wall... destroy... else..."
            # My `remove_wall` logic is for when we KNOW we want to destroy it.
            pass

    # Let's rewrite the loop logic clearly:
    # We need a way to check if a wall exists. 
    # A wall exists at (r, c) if it hasn't been destroyed yet.
    # We can use a set `destroyed_walls`.
    
    # Since I cannot redefine functions inside the loop, I'll use a comprehension 
    # or a map, but a for loop is allowed. I just need to avoid the `def` 
    # inside the loop or use a different structure.
    
    # Let's use a set for destroyed walls and the sorted lists for searching.
    # To avoid the `def` issue, I'll use a helper function defined outside.
    
def process_queries(H, W, Q, queries):
    # row_walls[r] = sorted list of column indices of existing walls
    # col_walls[c] = sorted list of row indices of existing walls
    row_walls = [list(range(1, W + 1)) for _ in range(H)]
    col_walls = [list(range(1, H + 1)) for _ in range(W)]
    
    # Set of (r, c) that are destroyed
    destroyed = set()
    
    # We need a function to remove a wall. Since we can't define it inside,
    # we use a logic block. But we need to call it multiple times.
    # We can use a list of functions or a custom class.
    
    class WallManager:
        def __init__(self, h, w):
            self.row_walls = [list(range(1, w + 1)) for _ in range(h)]
            self.col_walls = [list(range(1, h + 1)) for _ in range(w)]
            self.destroyed = set()
            
        def remove(self, r, c):
            if (r, c) in self.destroyed:
                return
            self.destroyed.add((r, c))
            # Remove from row_walls[r-1]
            idx_c = bisect_left(self.row_walls[r-1], c)
            self.row_walls[r-1].pop(idx_c)
            # Remove from col_walls[c-1]
            idx_r = bisect_left(self.col_walls[c-1], r)
            self.col_walls[c-1].pop(idx_r)

        def query(self, r, c):
            if (r, c) not in self.destroyed:
                self.remove(r, c)
            else:
                # Look up
                idx = bisect_left(self.col_walls[c-1], r)
                if idx > 0:
                    self.remove(self.col_walls[c-1][idx-1], c)
                # Look down
                if idx < len(self.col_walls[c-1]):
                    self.remove(self.col_walls[c-1][idx], c)
                # Look left
                idx = bisect_left(self.row_walls[r-1], c)
                if idx > 0:
                    self.remove(r, self.row_walls[r-1][idx-1])
                # Look right
                if idx < len(self.row_walls[r-1]):
                    self.remove(r, self.row_walls[r-1][idx])

    manager = WallManager(H, W)
    for r, c in queries:
        manager.query(r, c)
    
    return (H * W) - len(manager.destroyed)

# To glue this together into the required format:
import sys
from bisect import bisect_left

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = [(int(input_data[i]), int(input_data[i+1])) for i in range(3, len(input_data), 2)]
    
    class WallManager:
        def __init__(self, h, w):
            self.row_walls = [list(range(1, w + 1)) for _ in range(h)]
            self.col_walls = [list(range(1, h + 1)) for _ in range(w)]
            self.destroyed = set()
            
        def remove(self, r, c):
            if (r, c) in self.destroyed:
                return
            self.destroyed.add((r, c))
            idx_c = bisect_left(self.row_walls[r-1], c)
            self.row_walls[r-1].pop(idx_c)
            idx_r = bisect_left(self.col_walls[c-1], r)
            self.col_walls[c-1].pop(idx_r)

        def query(self, r, c):
            if (r, c) not in self.destroyed:
                self.remove(r, c)
            else:
                # Up/Down
                idx_col = bisect_left(self.col_walls[c-1], r)
                # We must capture the targets before removing them to avoid index shifts
                # though we are using a set to guard remove(), the indices in the 
                # current list change. However, we are looking in DIFFERENT lists
                # (one for row, one for col). The only issue is if we remove 
                # the same wall twice. The `remove` method handles that.
                
                # To be safe, identify all 4 targets first.
                targets = []
                if idx_col > 0:
                    targets.append((self.col_walls[c-1][idx_col-1], c))
                if idx_col < len(self.col_walls[c-1]):
                    targets.append((self.col_walls[c-1][idx_col], c))
                
                idx_row = bisect_left(self.row_walls[r-1], c)
                if idx_row > 0:
                    targets.append((r, self.row_walls[r-1][idx_row-1]))
                if idx_row < len(self.row_walls[r-1]):
                    