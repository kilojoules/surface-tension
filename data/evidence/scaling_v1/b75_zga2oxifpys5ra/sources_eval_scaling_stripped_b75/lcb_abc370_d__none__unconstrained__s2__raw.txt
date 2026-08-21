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
    queries = input_data[3:]
    
    # We need to track existing walls in each row and each column.
    # Using a list of sorted lists (acting as sets) to find the nearest wall.
    # Since H*W is up to 4e5, we can't use a full 2D array for wall status.
    # However, we can use a set of (r, c) tuples to check if a wall exists.
    
    # To efficiently find the nearest wall, we maintain sorted lists of 
    # existing wall indices for every row and every column.
    # Initial state: every cell has a wall.
    # But initializing H lists of size W and W lists of size H is O(H*W).
    # We use a trick: instead of lists, we use a data structure that allows
    # efficient deletion and successor/predecessor search.
    # In Python, since we can't use std::set, and H*W is 4e5, 
    # we can use the 'bisect' module on sorted lists. 
    # Deleting from a list is O(N), which is too slow.
    # However, we can use a Fenwick tree or Segment tree to find the nearest 1.
    # But the simplest way to implement "find nearest" without loops is 
    # using a SortedList from sortedcontainers, but that's not standard.
    # Given the constraints and Python, we must avoid O(N) deletions.
    
    # Let's use a different approach: 
    # We can't use loops, but we can use recursion or map/filter.
    # Actually, the most efficient way to handle "nearest" in Python 
    # without external libraries is using a Segment Tree or Fenwick Tree
    # implemented iteratively.
    
    # But wait, the problem says H*W <= 4e5. We can afford O(H*W) space.
    # We can use a 1D array to represent the grid: index = (r-1)*W + (c-1).
    # To find the nearest wall, we can't loop. 
    # Let's use the fact that we can use 'set' for O(1) check and 
    # 'bisect' on sorted lists. To avoid O(N) deletion, we can use 
    # a technique called "Square Root Decomposition" or just use 
    # the fact that we can rebuild the sorted lists periodically.
    # Or, more simply, use a Fenwick tree to find the nearest wall.
    # A Fenwick tree can tell us if a range has any walls. 
    # Binary search over the Fenwick tree finds the index.
    
    # However, implementing a full 2D-like structure with Fenwick trees 
    # in Python without loops is challenging.
    # Let's use the property that we can use `set` and `bisect` 
    # if we use a `SortedList` implementation. Since we can't import it,
    # we can implement a simple one using a list of blocks (sqrt decomposition).
    
    class SortedList:
        def __init__(self, iterable):
            self._len = sum(map(len, (self._build(sorted(iterable)))))
            self._blocks = self._build(sorted(iterable))
            self._size = 700 # Approx sqrt(4e5)

        def _build(self, sorted_list):
            return [sorted_list[i:i + self._size] 
                    for i in range(0, len(sorted_list), self._size)]

        def remove(self, val):
            idx = bisect_left(self._blocks, val, key=lambda b: b[-1] if b else float('-inf'))
            if idx < len(self._blocks):
                b = self._blocks[idx]
                i = bisect_left(b, val)
                if i < len(b) and b[i] == val:
                    b.pop(i)
                    self._len -= 1
                    if not b: self._blocks.pop(idx)
                    return True
            return False

        def find_nearest(self, val, direction):
            # direction: -1 for left/up, 1 for right/down
            # This is tricky without loops. Let's use a helper.
            return self._search(val, direction)

        def _search(self, val, direction):
            # Find index of first element > val (for direction 1) 
            # or first element < val (for direction -1)
            if direction == 1:
                idx = bisect_left(self._blocks, val, key=lambda b: b[-1] if b else float('-inf'))
                # We need the first element >= val. 
                # Check current block, then next.
                res = self._get_next(idx, val)
                return res
            else:
                idx = bisect_left(self._blocks, val, key=lambda b: b[0] if b else float('inf')) - 1
                res = self._get_prev(idx, val)
                return res

        def _get_next(self, idx, val):
            if idx >= len(self._blocks): return None
            b = self._blocks[idx]
            i = bisect_left(b, val)
            if i < len(b):
                # If we are looking for the first wall, and the current cell 
                # is already destroyed, we look for the first index > val.
                # The logic in the problem: if wall at (Rq, Cq) exists, destroy it.
                # If not, destroy nearest.
                # This means the 'val' itself is already gone.
                # So we look for the first element > val.
                # Since we use a set-like structure, we search for the first element > val.
                pass
            # To avoid loops, we use a conditional expression.
            # But we need to handle the case where the first element in the block is the val.
            # Since the wall at (Rq, Cq) is already gone, we search for > val.
            i = bisect_left(b, val + 1)
            if i < len(b): return b[i]
            if idx + 1 < len(self._blocks): return self._blocks[idx+1][0]
            return None

        def _get_prev(self, idx, val):
            if idx < 0: return None
            b = self._blocks[idx]
            i = bisect_left(b, val) - 1
            if i >= 0: return b[i]
            if idx - 1 >= 0: return self._blocks[idx-1][-1]
            return None

    # To avoid the complexity of the SortedList, let's use the fact that 
    # we can use a 1D array and a custom function to simulate the process.
    # But we can't use loops. We can use a recursive-like structure with 
    # a list comprehension or map, but that's basically a loop.
    # The only way to truly avoid loops is to use built-ins.
    
    # Let's use the SortedList logic but implemented without explicit for/while.
    # We can use a list of sets or similar, but we need the nearest element.
    # Actually, the most Pythonic way to do this without loops is to use 
    # a dictionary to store the walls and perform the queries.
    # But we still need to find the nearest wall.
    
    # Given the constraints and the "no loop" rule, the only way to find 
    # the nearest element in a sorted structure is bisect.
    # To handle deletions in O(sqrt N), we can use a list of lists.
    
    # Let's refine the SortedList to be completely loop-free.
    def get_next(blocks, val):
        idx = bisect_left(blocks, val, key=lambda b: b[-1] if b else float('-inf'))
        if idx >= len(blocks): return None
        b = blocks[idx]
        i = bisect_left(b, val + 1)
        if i < len(b): return b[i]
        return blocks[idx+1][0] if idx + 1 < len(blocks) else None

    def get_prev(blocks, val):
        idx = bisect_left(blocks, val, key=lambda b: b[0] if b else float('inf')) - 1
        if idx < 0: return None
        b = blocks[idx]
        i = bisect_left(b, val) - 1
        if i >= 0: return b[i]
        return blocks[idx-1][-1] if idx - 1 >= 0 else None

    # Since we can't use loops to initialize the blocks, we use list comprehensions.
    # But we can't use loops to process queries. 
    # We can use `functools.reduce` to process queries.
    from functools import reduce

    def remove_val(blocks, val):
        idx = bisect_left(blocks, val, key=lambda b: b[-1] if b else float('-inf'))
        if idx < len(blocks):
            b = blocks[idx]
            i = bisect_left(b, val)
            if i < len(b) and b[i] == val:
                # Create a new block without the value
                new_b = b[:i] + b[i+1:]
                # Create a new blocks list
                # If the block becomes empty, remove it
                if not new_b:
                    return blocks[:idx] + blocks[idx+1:]
                return blocks[:idx] + [new_b] + blocks[idx+1:]
        return blocks

    # Initialize blocks for each row and column
    # H*W is 4e5, so we can't have H lists of size W.
    # Wait, the constraint is H*W <= 4e5. That means we CAN have 
    # a list of lists, as long as the total number of elements is 4e5.
    
    # To avoid loops in initialization:
    # row_blocks = [ [list(range(1, W+1))[i:i+700] for i in range(0, W, 700)] for _ in range(H) ]
    # But the above is a nested comprehension (loop). 
    # The prompt says "no for/while loops". Comprehensions are explicitly allowed.
    
    # Let's use a simpler structure: a set of walls and 
    # for each row/col, a sorted list. 
    # Since we can't use loops to update, we use reduce.
    
    # To make it efficient, we use a dictionary of sorted lists.
    # But deleting from a list is O(W). With W=4e5, that's O(Q*W).
    # We MUST use the block-based SortedList or a similar structure.
    
    # Let's use the reduce pattern to process queries.
    # state = (row_blocks, col_blocks, wall_set)
    
    def process_query(state, query):
        row_blocks, col_blocks, wall_set = state
        r, c = map(int, query)
        
        if (r, c) in wall_set:
            # Destroy wall at (r, c)
            new_row_blocks = list(row_blocks)
            new_row_blocks[r-1] = remove_val(row_blocks[r-1], c)
            new_col_blocks = list(col_blocks)
            new_col_blocks[c-1] = remove_val(col_blocks[c-1], r)
            return (new_row_blocks, new_col_blocks, wall_set - {(r, c)})
        else:
            # Destroy 4 nearest walls
            # Find targets
            up = get_prev(col_blocks[c-1], r)
            down = get_next(col_blocks[c-1], r)
            left = get_prev(row_blocks[r-1], c)
            right = get_next(row_blocks[r-1], c)
            
            targets = [t for t in [(up, c), (down, c), (r, left), (r, right)] if t[0] and t[1]]
            # Filter out None and ensure coordinates are valid
            targets = [t for t in targets if t[0] is not None and t[1] is not None]
            
            # Update state by removing all targets
            # We can use reduce again to remove each target from the state
            return reduce(
                lambda