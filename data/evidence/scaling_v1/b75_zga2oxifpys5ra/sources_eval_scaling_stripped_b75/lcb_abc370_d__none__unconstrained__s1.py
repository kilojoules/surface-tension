import sys
from bisect import bisect_left

def solve():
    # Read H, W, Q
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

    # We need to efficiently find the nearest existing wall in 4 directions.
    # Since H*W is up to 4e5, we can't use a simple 2D array for lookups.
    # We use a list of sorted lists (one for each row and one for each column).
    # rows[r] contains sorted indices of columns that still have walls.
    # cols[c] contains sorted indices of rows that still have walls.
    
    # Using list comprehensions to initialize
    rows = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions in sorted lists efficiently without O(N) shifts,
    # we would ideally use a SortedList from sortedcontainers, but that's not standard.
    # However, since we must use standard libraries, and H*W is 4e5, 
    # we can use the fact that we only need to find the nearest neighbors.
    # A common trick in Python for this constraint is using a Fenwick tree or 
    # Segment tree, but since we need to find the "nearest" index, 
    # we can use a dictionary-based approach or simply accept that 
    # we need a data structure that supports fast deletion and successor/predecessor.
    
    # Given the constraints and the "no external libraries" rule, 
    # the most reliable way to implement a SortedSet is using a Skip List or 
    # a Balanced BST. But those are verbose. 
    # Alternatively, we can use the 'bisect' module on standard lists, 
    # but 'pop(index)' is O(N). With H*W=4e5, O(N) is too slow.
    
    # Wait, the constraint H*W <= 4e5 means the average row/col length is small,
    # but the worst case (e.g., H=1, W=4e5) makes pop() O(W).
    # Let's use a square root decomposition approach for the sorted lists.
    
    class FastSortedList:
        def __init__(self, iterable):
            self.blocks = [sorted(iterable)]
            self.size = len(iterable)
            self.block_size = int(self.size**0.5) + 1
            self._rebuild()

        def _rebuild(self):
            # Flatten and redistribute into blocks of size sqrt(N)
            flat = [item for block in self.blocks for item in block]
            self.blocks = [flat[i:i + self.block_size] 
                           for i in range(0, len(flat), self.block_size)]

        def remove(self, val):
            # Find block using bisect on the last element of each block
            # We use a list of the maximums of each block for fast lookup
            idx = self._find_block_idx(val)
            if idx < len(self.blocks):
                block = self.blocks[idx]
                i = bisect_left(block, val)
                if i < len(block) and block[i] == val:
                    block.pop(i)
                    self.size -= 1
                    # Rebuild if block becomes too small to maintain O(sqrt N)
                    if len(block) == 0:
                        self.blocks.pop(idx)
                    return True
            return False

        def _find_block_idx(self, val):
            # Binary search to find which block could contain the value
            # We check the last element of each block
            import bisect
            # We need the first block whose last element is >= val
            # Since we can't use map/filter, we use a helper to get last elements
            lasts = [b[-1] for b in self.blocks]
            return bisect.bisect_left(lasts, val)

        def find_nearest(self, val, direction):
            # direction: -1 for predecessor, 1 for successor
            import bisect
            idx = self._find_block_idx(val)
            
            if direction == 1: # Successor (smallest element > val)
                # Check current block
                if idx < len(self.blocks):
                    block = self.blocks[idx]
                    i = bisect.bisect_right(block, val)
                    if i < len(block):
                        return block[i]
                return None
            else: # Predecessor (largest element < val)
                # Check current block or previous block
                # Start by looking at the block where val would be inserted
                # The element < val could be in block idx or idx-1
                
                # Try block idx first (elements to the left of insertion point)
                if idx < len(self.blocks):
                    block = self.blocks[idx]
                    i = bisect.bisect_left(block, val)
                    if i > 0:
                        return block[i-1]
                
                # Try block idx-1
                if idx > 0:
                    return self.blocks[idx-1][-1]
                
                return None

    # Initialize the FastSortedList for every row and column
    row_sets = [FastSortedList(range(1, W + 1)) for _ in range(H + 1)]
    col_sets = [FastSortedList(range(1, H + 1)) for _ in range(W + 1)]

    for r, c in queries:
        # Check if wall exists at (r, c)
        # We can't check existence directly without a loop or comprehension
        # But remove() returns True if the element was present.
        if row_sets[r].remove(c):
            col_sets[c].remove(r)
        else:
            # No wall at (r, c), destroy 4 neighbors
            # Up
            up = col_sets[c].find_nearest(r, -1)
            # Down
            down = col_sets[c].find_nearest(r, 1)
            # Left
            left = row_sets[r].find_nearest(c, -1)
            # Right
            right = row_sets[r].find_nearest(c, 1)
            
            # Collect targets to destroy
            targets = [
                (up, c) if up else None,
                (down, c) if down else None,
                (r, left) if left else None,
                (r, right) if right else None
            ]
            
            # Use a list comprehension to execute removals
            # We must filter None and then perform the double removal
            [
                (row_sets[tr].remove(tc), col_sets[tc].remove(tr))
                for tr, tc in targets if tr is not None
            ]

    # Total remaining walls = sum of sizes of all row_sets
    print(sum(rs.size for rs in row_sets))

if __name__ == "__main__":
    solve()