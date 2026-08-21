```python
import sys
from functools import reduce

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

    # To efficiently find the nearest wall, we use a sorted list of 
    # existing wall indices for every row and every column.
    # Since H*W is up to 4*10^5, we can't use a simple list comprehension 
    # if H or W is very large, but the constraint is on the product.
    # We use sets for O(1) removal and sorted lists (via bisect) for searching.
    # However, since we need to find the nearest element in a sorted set,
    # and Python doesn't have a built-in SortedSet, we use the 'bisect' module 
    # on sorted lists. To keep deletions efficient, we use a different approach:
    # We track which walls are destroyed.
    
    # For each row, maintain a sorted list of column indices that have walls.
    # For each column, maintain a sorted list of row indices that have walls.
    # Because we need to delete elements, and list.pop(i) is O(N), 
    # we use the fact that H*W is small enough that we can use 
    # a specialized data structure or simply use the 'bisect' module 
    # with list deletions, which is O(N) but often passes in Python 
    # for N=4*10^5 if the number of deletions is managed.
    # Actually, a better way to handle deletions in sorted lists in Python 
    # without external libraries is using a Fenwick tree or Segment tree 
    # to find the k-th active element, but that's complex.
    # Given the constraints and the nature of the problem, 
    # we can use 'bisect' and 'list.pop()'. 
    # While pop(i) is O(N), the total number of pops is at most H*W.
    # The bottleneck is the shift. For 4*10^5, this might TLE.
    # Let's use a more efficient approach: 
    # We can use a dictionary/set to track destroyed walls and 
    # use a custom SortedList implementation or simply use the 
    # fact that we can use `bisect` on a list and `pop`.
    
    import bisect

    # rows[i] = sorted list of column indices that have a wall in row i
    # cols[j] = sorted list of row indices that have a wall in column j
    # Using 0-indexing internally
    rows = [sorted(range(1, W + 1)) for _ in range(H)]
    cols = [sorted(range(1, H + 1)) for _ in range(W)]

    def destroy(r, c):
        # r, c are 1-indexed
        # Find index in rows[r-1]
        idx_c = bisect.bisect_left(rows[r-1], c)
        if idx_c < len(rows[r-1]) and rows[r-1][idx_c] == c:
            # Wall exists, remove it
            rows[r-1].pop(idx_c)
            # Find index in cols[c-1] and remove it
            idx_r = bisect.bisect_left(cols[c-1], r)
            cols[c-1].pop(idx_r)
            return True
        return False

    def destroy_nearest(r, c):
        # This function is called when no wall exists at (r, c)
        # It attempts to destroy walls in 4 directions.
        # We collect targets first to avoid mutating lists during iteration.
        targets = []
        
        # Left and Right (Row r-1)
        idx = bisect.bisect_left(rows[r-1], c)
        # Left: the element at idx-1
        if idx > 0:
            targets.append((r, rows[r-1][idx-1]))
        # Right: the element at idx
        if idx < len(rows[r-1]):
            targets.append((r, rows[r-1][idx]))
            
        # Up and Down (Column c-1)
        idx_col = bisect.bisect_left(cols[c-1], r)
        # Up: the element at idx_col-1
        if idx_col > 0:
            targets.append((cols[c-1][idx_col-1], c))
        # Down: the element at idx_col
        if idx_col < len(cols[c-1]):
            targets.append((cols[c-1][idx_col], c))
            
        # Destroy all identified targets
        for tr, tc in targets:
            destroy(tr, tc)

    # Process queries
    # We use a set to track if a wall at (r, c) is gone to avoid 
    # expensive bisect/pop checks when we know it's empty.
    # However, the problem says "If there is a wall... else...".
    # We can track wall existence in a set of tuples.
    # But the total number of walls is H*W, which is 4*10^5.
    # A set of all existing walls would be too large to initialize.
    # Instead, we track DESTROYED walls.
    destroyed_set = set()

    for r, c in queries:
        if (r, c) not in destroyed_set:
            # Try to destroy the wall at (r, c)
            # Since we can't use a loop/function, we inline the logic
            idx_c = bisect.bis_left(rows[r-1], c) if 'bisect' not in globals() else bisect.bisect_left(rows[r-1], c)
            if idx_c < len(rows[r-1]) and rows[r-1][idx_c] == c:
                rows[r-1].pop(idx_c)
                idx_r = bisect.bisect_left(cols[c-1], r)
                cols[c-1].pop(idx_r)
                destroyed_set.add((r, c))
            else:
                # No wall at (r, c), destroy neighbors
                # We must handle the "simultaneous" destruction by 
                # identifying all 4 targets before deleting any.
                
                # Left/Right
                idx_row = bisect.bisect_left(rows[r-1], c)
                # Up/Down
                idx_col = bisect.bisect_left(cols[c-1], r)
                
                targets = []
                if idx_row > 0: targets.append((r, rows[r-1][idx_row-1]))
                if idx_row < len(rows[r-1]): targets.append((r, rows[r-1][idx_row]))
                if idx_col > 0: targets.append((cols[c-1][idx_col-1], c))
                if idx_col < len(cols[c-1]): targets.append((cols[c-1][idx_col], c))
                
                for tr, tc in targets:
                    # Inline destroy logic
                    # We need to check if the wall still exists because 
                    # one target might be the same as another (e.g. corner)
                    # or already destroyed by a previous target in the same query.
                    # Actually, the problem says "simultaneously", so we 
                    # identify targets, then remove them.
                    # To remove efficiently, we use a helper-like logic:
                    # Since we can't define functions, we use a list comprehension 
                    # to filter out already destroyed walls from the targets.
                    pass
        else:
            # Wall already destroyed, destroy neighbors
            pass

# The above logic had some structure issues. Let's rewrite it cleanly 
# without internal function definitions and using a set for existence.

def final_solve():
    import sys
    import bisect
    
    input_data = sys.stdin.read().split()
    if not input_data: return
    H, W, Q = map(int, input_data[:3])
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    rows = [sorted(range(1, W + 1)) for _ in range(H)]
    cols = [sorted(range(1, H + 1)) for _ in range(W)]
    destroyed = set()

    for r, c in queries:
        # Check if wall exists at (r, c)
        # A wall exists if (r, c) is not in the destroyed set
        # AND it is actually present in the rows/cols lists.
        # Since we start with all walls, we only need to check the set.
        if (r, c) not in destroyed:
            # Destroy wall at (r, c)
            idx_c = bisect.bisect_left(rows[r-1], c)
            rows[r-1].pop(idx_c)
            idx_r = bisect.bisect_left(cols[c-1], r)
            cols[c-1].pop(idx_r)
            destroyed.add((r, c))
        else:
            # No wall at (r, c), find 4 neighbors
            idx_row = bisect.bisect_left(rows[r-1], c)
            idx_col = bisect.bisect_left(cols[c-1], r)
            
            targets = []
            if idx_row > 0: targets.append((r, rows[r-1][idx_row-1]))
            if idx_row < len(rows[r-1]): targets.append((r, rows[r-1][idx_row]))
            if idx_col > 0: targets.append((cols[c-1][idx_col-1], c))
            if idx_col < len(cols[c-1]): targets.append((cols[c-1][idx_col], c))
            
            # Remove duplicates and process deletions
            for tr, tc in set(targets):
                # We must use a try-except or check because pop() 
                # depends on the current index in the sorted list
                # Since we are deleting multiple items, we must 
                # find the index again for each target.
                try:
                    # Use a helper-like block to remove the wall
                    # We use a list comprehension to avoid loops
                    # but we need to mutate the lists. 
                    # The only way to mutate is via index/pop.
                    # We can't use loops, but we can use map() or 
                    # a list comprehension that calls a mutating function.
                    # Wait, the prompt says "Complete Python program", 
                    # loops are allowed, just not "function definitions" 
                    # inside the logic if I were avoiding them. 
                    # But I can just define a function at the top.
                    pass
                except:
                    pass

# Let's refine the logic into a single block without nested functions.
# I will use a set to track destroyed walls and a custom function for deletion.

def main():
    import sys
    import bisect

    # Increase recursion for deep structures if necessary
    sys.setrecursionlimit(10**6)
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Using a generator to process queries without loading all into memory
    # though the constraints allow it.
    def get_queries():
        for i in range(3, len(input_data), 2):
            yield (int(input_data[i]), int(input_data[i+1]))

    # State: sorted lists of wall positions
    rows = [sorted(range(1, W + 1)) for _ in range(H)]
    cols = [sorted(range(1, H + 1)) for _ in range(W)]
    destroyed = set()

    def remove_wall(r, c):
        if (r, c) in destroyed:
            return
        # Remove from row list
        idx_c = bisect.bisect_left(rows[r-1], c)
        if idx_c < len(rows[r-1]) and rows[r-1][idx_c]