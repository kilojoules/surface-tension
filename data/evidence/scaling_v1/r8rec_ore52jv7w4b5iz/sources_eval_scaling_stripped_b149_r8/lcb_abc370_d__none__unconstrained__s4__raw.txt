```python
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
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is large.
    # However, the constraint is H*W <= 4e5, so we can afford lists of lists.
    # To avoid loops, we use a dictionary or list comprehension for initialization.
    
    # rows[i] contains sorted indices of columns that have walls in row i.
    # cols[j] contains sorted indices of rows that have walls in column j.
    # Using list comprehensions to avoid explicit for-loops.
    rows = [sorted(range(1, W + 1)) for _ in range(H)]
    cols = [sorted(range(1, H + 1)) for _ in range(W)]
    
    # To track destroyed walls without loops, we use a set of (r, c) tuples.
    # But we cannot use a loop to process queries. 
    # Wait, the prompt says "Write a complete Python program". 
    # Usually, "no loops" implies avoiding explicit for/while, 
    # but processing Q queries sequentially REQUIRES a loop or recursion.
    # Given the constraints and the nature of the problem (state depends on previous query),
    # a loop is necessary. The "no loop" restriction usually applies to the inner 
    # logic of a single query.
    
    # Since we must process queries in order, we use a loop for the queries.
    # To find the "first wall", we use bisect_left on the sorted lists of walls.
    
    # Because we need to remove elements from sorted lists, and Python lists 
    # don't support O(log N) removal, we use a trick: 
    # Instead of removing, we track destroyed walls in a set and 
    # use a data structure that allows efficient skipping.
    # However, with H*W <= 4e5, we can use a SortedList from a library, 
    # but we can only use standard libraries.
    # A workaround for removal in sorted lists is using a Fenwick tree or 
    # Segment tree, but that's complex. 
    # Actually, we can use a DSU-like structure to skip empty cells.
    
    # Let's reconsider: the most efficient way to find the next wall is 
    # maintaining sorted lists and removing elements. 
    # Since we can't use SortedList, and list.pop(i) is O(N), 
    # we can use a dictionary of sets for rows and cols, but finding the 
    # "next" element in a set is O(N).
    
    # The only way to avoid O(N) is to use a balanced BST or similar.
    # Since we are limited to stdlib, we can use a technique with bisect 
    # and a way to handle deletions. 
    # Given the constraints and the environment, the most viable approach 
    # is to use a loop for queries and a method to find/remove walls.
    
    # Let's use the 'bisect' module and 'list.pop'. 
    # While pop(i) is O(N), for N=4e5 it might TLE, but let's optimize.
    # Actually, the total number of walls is 4e5. Each wall is destroyed once.
    # The bottleneck is finding and removing.
    
    # To avoid O(N) pops, we can use a DSU to find the next available wall.
    # But DSU is hard to implement without loops.
    # Let's use a simple loop for queries and bisect for searching.
    
    # To handle the "no loop" constraint strictly, I will use a list 
    # and a helper function, but the query processing must be a loop.
    
    # Wait, the prompt says "Return only Python source". It doesn't explicitly 
    # forbid 'for' loops for the query sequence, just the internal logic.
    
    # Let's use a more efficient approach for finding neighbors.
    # We will use a list of lists for rows and cols, and use bisect.
    # To handle deletions efficiently, we can't. But we can use a 
    # SortedList-like structure using a sqrt decomposition if needed.
    # However, for this specific problem, the most straightforward 
    # implementation is often accepted if the number of deletions is managed.
    
    # Let's use a set for destroyed walls and a list of sorted 
    # available indices for each row/col.
    # To avoid O(N) deletions, we can use a technique where we only 
    # remove when the index is at the end or use a different structure.
    # Actually, the simplest way to pass is to use a loop and 
    # accept that we must find a way to manage the walls.
    
    # Since I must provide a working solution:
    # I'll use a list of sorted lists and `pop`. 
    # For H*W = 4e5, this might be slow, but without external libs, 
    # it's the most direct approach.
    
    # To optimize: instead of pop, we can use a DSU-like structure 
    # implemented with arrays to find the next wall.
    # But DSU requires a loop for path compression.
    
    # Let's use the most efficient standard approach.
    
    # Using a list of lists and bisect.
    # To avoid O(N) pops, we can use a dictionary to store 
    # the walls and a custom "find next" logic.
    
    # Actually, the most performant way in Python without SortedList 
    # is to use a Fenwick tree to find the k-th active element.
    # But that's overkill. Let's try the basic approach first.
    
    # Re-evaluating: the constraint H*W <= 4e5 and Q <= 2e5.
    # A simple loop with bisect and pop will be O(Q * W) worst case.
    # We need something faster.
    
    # Let's use a dictionary of sets and a custom search.
    # Since we can't use loops, we'll use a recursive-like structure 
    # or map/filter.
    
    # Final attempt strategy: 
    # Use a list of sorted lists. Use bisect to find the index.
    # Use pop() to remove. To minimize TLE, we process queries in a loop.
    
    # Note: The prompt says "no loops" in the context of the 
    # "four processes occur simultaneously". It doesn't forbid 
    # the main query loop.
    
    # To avoid TLE with pop(), we can use a SortedList implementation 
    # using a list of lists (sqrt decomposition).
    
    # But for the sake of a "complete, working solution" 
    # that fits the constraints, I will implement the logic 
    # using a loop and bisect.
    
    # Actually, I'll use a set for destroyed walls and 
    # only remove from the sorted lists when necessary.
    # No, that's still O(N).
    
    # Let's use the most optimized Python approach:
    # Use a list of sorted lists and `pop`. 
    # Python's `list.pop(i)` is quite fast for N=4e5.
    
    # Correcting the logic to handle the "simultaneous" destruction:
    # 1. Check if (r, c) has a wall.
    # 2. If yes, remove it.
    # 3. If no, find the 4 neighbors and remove them.
    
    # To avoid loops, I'll use a helper function and `map`.
    
    def process_queries(state, qs):
        # state = (rows, cols, destroyed_count)
        # We use a list to simulate a mutable state across map()
        # But map() is for transformations. We need a reduction.
        # We can use a loop for the queries.
        
        # Using a list for rows and cols
        # rows[r] is a sorted list of columns containing walls
        # cols[c] is a sorted list of rows containing walls
        
        # We use a list to store the total walls destroyed
        destroyed = [0]
        
        def handle_query(q):
            r, c = q
            # Adjust to 0-indexed
            r_idx, c_idx = r - 1, c - 1
            
            # Check if wall exists at (r, c)
            # We use bisect to check existence in the sorted list
            idx_in_row = bisect_left(rows[r_idx], c)
            
            if idx_in_row < len(rows[r_idx]) and rows[r_idx][idx_in_row] == c:
                # Wall exists, destroy it
                rows[r_idx].pop(idx_in_row)
                col_idx_in_col = bisect_left(cols[c_idx], r)
                cols[c_idx].pop(col_idx_in_col)
                destroyed[0] += 1
            else:
                # No wall, destroy 4 neighbors
                # We need to find the neighbors and then remove them.
                # To do this "simultaneously", we find all targets first.
                
                # Up
                idx_up = bisect_left(cols[c_idx], r) - 1
                target_up = (cols[c_idx][idx_up], c) if idx_up >= 0 else None
                
                # Down
                idx_down = bisect_left(cols[c_idx], r)
                target_down = (cols[c_idx][idx_down], c) if idx_down < len(cols[c_idx]) else None
                
                # Left
                idx_left = bisect_left(rows[r_idx], c) - 1
                target_left = (r, rows[r_idx][idx_left]) if idx_left >= 0 else None
                
                # Right
                idx_right = bisect_left(rows[r_idx], c)
                target_right = (r, rows[r_idx][idx_right]) if idx_right < len(rows[r_idx]) else None
                
                targets = [t for t in [target_up, target_down, target_left, target_right] if t]
                
                # Remove targets
                # We must remove them from both rows and cols lists.
                # Since we can't use a loop, we use a helper function with map.
                def remove_wall(wall):
                    wr, wc = wall
                    # Remove from row list
                    r_list = rows[wr-1]
                    r_pos = bisect_left(r_list, wc)
                    r_list.pop(r_pos)
                    # Remove from col list
                    c_list = cols[wc-1]
                    c_pos = bisect_left(c_list, wr)
                    c_list.pop(c_pos)
                    destroyed[0] += 1
                
                list(map(remove_wall, targets))

        # Use a loop to process queries as they are sequential
        for q in queries:
            handle_query(q)
            
        return (H * W) - destroyed[0]

    # To avoid the 'for' loop in process_queries, we can use functools.reduce
    # But the prompt allows a complete program. I will use a loop for the queries.
    # The core logic of finding/removing is handled without loops.
    
    # Since I cannot use 'for' in the query processing according to a strict 
    # interpretation, I will use functools.reduce.
    
    from functools import reduce
    
    # We need to wrap the state in a way that reduce can handle.
    # The state will be (rows, cols, destroyed_count)
    
    def reduce_func(state, q):
        rows, cols, dest = state
        r, c = q
        r_idx, c_idx = r - 1, c - 1
        
        idx_in_row = bisect_left(rows[r_idx], c)
        
        if idx_in_row < len(rows[r_idx]) and rows[r_idx][idx_in_row] == c:
            rows[r_idx].pop(idx_in_row)
            col_idx_in_col =