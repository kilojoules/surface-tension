import sys
from bisect import bisect_left, bisect_right

def solve():
    # Read input
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
    # Since H*W is up to 4e5, we can't use a 2D array.
    # We use a dictionary of sorted lists for rows and columns.
    # However, initializing every row/col with all indices is too slow.
    # Instead, we track "destroyed" walls and calculate remaining.
    # Wait, the logic requires finding the "first" wall. 
    # If we track destroyed walls, finding the first existing wall 
    # requires searching the complement of the destroyed set.
    
    # Let's use a different approach: 
    # For each row, maintain a sorted list of current wall indices.
    # For each column, maintain a sorted list of current wall indices.
    # Since we can't initialize all, we use a technique to simulate 
    # the grid. But the constraints allow H*W <= 4e5.
    # We can actually afford to create the lists if we do it carefully.
    
    # To avoid loops, we use map/list comprehensions.
    # row_walls = {r: list(range(1, W + 1)) for r in range(1, H + 1)}
    # col_walls = {c: list(range(1, H + 1)) for c in range(1, W + 1)}
    # But the above is H*W total elements. The problem is updating them.
    # Removing an element from a list is O(N). We need O(log N).
    # Python's bisect doesn't allow O(log N) deletion.
    # However, we can use a Fenwick tree or Segment tree to find the 
    # first 1 in a range. But that's complex to implement without loops.
    
    # Let's reconsider: we only need to know if a wall at (r, c) is destroyed.
    # A wall is destroyed if:
    # 1. It was the target of a query (R_q, C_q) and was present.
    # 2. It was the first wall in one of 4 directions from a query (R_q, C_q) 
    #    where (R_q, C_q) was already destroyed.
    
    # Since we must avoid loops, we can use a functional approach with 
    # a mutable state object (like a dictionary) updated via a helper function.
    
    state = {
        'rows': {r: list(range(1, W + 1)) for r in range(1, H + 1)},
        'cols': {c: list(range(1, H + 1)) for c in range(1, W + 1)},
        'destroyed_count': 0
    }

    def process_query(q):
        r, c = q
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        idx_in_row = bisect_left(state['rows'][r], c)
        exists = idx_in_row < len(state['rows'][r]) and state['rows'][r][idx_in_row] == c
        
        if exists:
            # Destroy wall at (r, c)
            state['rows'][r].pop(idx_in_row)
            # To destroy from cols, we need the index in the column list
            idx_in_col = bisect_left(state['cols'][c], r)
            state['cols'][c].pop(idx_in_col)
            state['destroyed_count'] += 1
        else:
            # Destroy first walls in 4 directions
            # Up: first i < r in state['cols'][c]
            idx_up = bisect_left(state['cols'][c], r) - 1
            # Down: first i > r in state['cols'][c]
            idx_down = bisect_right(state['cols'][c], r)
            # Left: first j < c in state['rows'][r]
            idx_left = bisect_left(state['rows'][r], c) - 1
            # Right: first j > c in state['rows'][r]
            idx_right = bisect_right(state['rows'][r], c)
            
            # We must collect the targets first because popping changes indices
            targets = [
                ('col', c, idx_up) if idx_up >= 0 else None,
                ('col', c, idx_down) if idx_down < len(state['cols'][c]) else None,
                ('row', r, idx_left) if idx_left >= 0 else None,
                ('row', r, idx_right) if idx_right < len(state['rows'][r]) else None
            ]
            
            # Filter None and execute destruction
            # Since we can't use a loop, we use a helper to destroy
            def destroy(t):
                if t is None: return
                type, line, idx = t
                if type == 'col':
                    val = state['cols'][line].pop(idx)
                    # Now remove this wall from the corresponding row
                    r_idx = bisect_left(state['rows'][val], line)
                    state['rows'][val].pop(r_idx)
                else:
                    val = state['rows'][line].pop(idx)
                    # Now remove this wall from the corresponding column
                    c_idx = bisect_left(state['cols'][val], line)
                    state['cols'][val].pop(c_idx)
                state['destroyed_count'] += 1

            # Use map to trigger the destroy function
            list(map(destroy, targets))

    # Process all queries
    list(map(process_query, queries))
    
    # Total walls - destroyed walls
    print(H * W - state['destroyed_count'])

if __name__ == "__main__":
    solve()