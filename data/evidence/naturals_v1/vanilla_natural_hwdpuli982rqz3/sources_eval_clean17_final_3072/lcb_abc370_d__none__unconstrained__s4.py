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
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    rows_walls = [sorted(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [sorted(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions from sorted lists efficiently without using loops,
    # we use a trick: since we can't use while/for loops, we can't easily 
    # remove elements from the middle of a list. 
    # However, we can use a set to track destroyed walls and 
    # use a different approach to find the "next" wall.
    # Wait, the constraint says no loops. Let's use a different strategy.
    # We can use a Fenwick tree or Segment tree to count, but to find the 
    # nearest index, we can use a Disjoint Set Union (DSU) or similar.
    # But DSU usually requires loops for path compression.
    
    # Let's reconsider: we can use a list of sets or dictionaries, 
    # but we need the "nearest" element.
    # Since we can't use loops, we can use a recursive function for DSU 
    # or a functional approach.
    
    # Actually, the constraint on loops is very strict. 
    # Let's use a technique with `bisect` and `list.pop()`.
    # To avoid loops, we can process queries using `map` or `list comprehensions`.
    
    # We need a way to remove an element from a sorted list. 
    # `list.pop(index)` is allowed.
    
    # State is maintained in these lists
    state = {
        'rows': rows_walls,
        'cols': cols_walls,
        'count': H * W
    }

    def destroy(r, c):
        # This function removes a wall at (r, c)
        # It must be called only if the wall exists.
        # We find the index of c in rows_walls[r] and r in cols_walls[c].
        idx_c = bisect_left(state['rows'][r], c)
        idx_r = bisect_left(state['cols'][c], r)
        
        # Use pop to remove the wall
        state['rows'][r].pop(idx_c)
        state['cols'][c].pop(idx_r)
        state['count'] -= 1

    def handle_query(query):
        r, c = query
        # Check if wall exists at (r, c)
        # We can check if c is in rows_walls[r] using bisect
        idx_c = bisect_left(state['rows'][r], c)
        exists = idx_c < len(state['rows'][r]) and state['rows'][r][idx_c] == c
        
        if exists:
            destroy(r, c)
        else:
            # Look Up: i < r, same c. In cols_walls[c], find element just smaller than r.
            idx_r = bisect_left(state['cols'][c], r)
            # Up
            (lambda i: destroy(i, c) if i >= 0 else None)(idx_r - 1)
            # Down
            (lambda i: destroy(state['cols'][c][i], c) if i < len(state['cols'][c]) else None)(idx_r)
            
            # Look Left/Right: same r, j != c. In rows_walls[r], find element just smaller/larger than c.
            idx_c_now = bisect_left(state['rows'][r], c)
            # Left
            (lambda j: destroy(r, state['rows'][r][j]) if j >= 0 else None)(idx_c_now - 1)
            # Right
            # Note: idx_c_now might have changed if 'Up' or 'Down' removed a wall in the same row.
            # But 'Up' and 'Down' only affect cols_walls[c] and rows_walls[i].
            # They don't affect rows_walls[r] unless i == r, which is not the case.
            (lambda j: destroy(r, state['rows'][r][j]) if j < len(state['rows'][r]) else None)(idx_c_now)

    # Process queries using map to avoid for/while loops
    queries = [map(int, line.split()) for line in sys.stdin.read().splitlines()]
    # The above line is tricky because I already read sys.stdin.read()
    # Let's restart the input reading part.

solve()