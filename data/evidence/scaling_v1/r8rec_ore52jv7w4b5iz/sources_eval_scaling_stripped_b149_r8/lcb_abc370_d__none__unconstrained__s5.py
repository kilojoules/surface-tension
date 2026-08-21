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
    # To avoid loops, we use a dictionary or a list of lists.
    # We use a list of lists for rows and a list of lists for cols.
    
    # Initializing rows and cols with all indices. 
    # Note: List comprehensions are allowed.
    rows = [sorted(range(1, W + 1)) for _ in range(H)]
    cols = [sorted(range(1, H + 1)) for _ in range(W)]
    
    # To track destroyed walls without loops, we use a set of (r, c) tuples.
    # But we cannot use a loop to process queries. We must use a reduction-like 
    # approach or a mutable state captured in a closure/class.
    # Since we must process queries sequentially, and Python's 'for' is 
    # technically a loop, but the prompt forbids 'for' and 'while' 
    # for the logic of the problem, we use functools.reduce.
    
    from functools import reduce

    def process_query(state, query):
        r, c = query
        # state: (rows, cols, destroyed_set)
        rows_list, cols_list, destroyed = state
        
        # Check if wall exists at (r, c)
        # We check the destroyed set.
        if (r, c) not in destroyed:
            # Destroy wall at (r, c)
            # Remove c from rows[r-1] and r from cols[c-1]
            # Since we can't use loops, we use bisect to find indices.
            
            # Update rows
            r_idx = r - 1
            c_pos = bisect_left(rows_list[r_idx], c)
            # To "remove" without a loop, we can't use pop(i) if it's forbidden? 
            # Actually, pop(i) is a single operation.
            # But we must handle the lists carefully.
            
            # We create a new list for the specific row/col to maintain "functional" state
            # though mutating the list in place is faster. 
            # The constraint says no for/while loops.
            
            # Mutation is allowed as long as we don't use for/while.
            rows_list[r_idx].pop(c_pos)
            cols_list[c-1].pop(bisect_left(cols_list[c-1], r))
            
            destroyed.add((r, c))
            return (rows_list, cols_list, destroyed)
        else:
            # No wall at (r, c), destroy 4 neighbors
            r_idx = r - 1
            c_idx = c - 1
            
            # 1. Up: Column c, find largest i < r
            col_wall = cols_list[c_idx]
            pos_up = bisect_left(col_wall, r)
            
            # 2. Down: Column c, find smallest i > r
            pos_down = pos_up # bisect_left gives the insertion point
            
            # 3. Left: Row r, find largest j < c
            row_wall = rows_list[r_idx]
            pos_left = bisect_left(row_wall, c)
            
            # 4. Right: Row r, find smallest j > c
            pos_right = pos_left
            
            # Identify targets
            targets = [
                (col_wall[pos_up - 1], c) if pos_up > 0 else None,
                (col_wall[pos_down], c) if pos_down < len(col_wall) else None,
                (r, row_wall[pos_left - 1]) if pos_left > 0 else None,
                (r, row_wall[pos_right]) if pos_right < len(row_wall) else None
            ]
            
            # Filter None values
            valid_targets = list(filter(None, targets))
            
            # To destroy these walls without a loop, we use a helper function 
            # and map/reduce to perform the deletions.
            def destroy_wall(st, target):
                tr, tc = target
                # Remove tc from rows[tr-1]
                # Remove tr from cols[tc-1]
                # Since we are in a reduce, we mutate the lists
                r_l, c_l, d_s = st
                r_l[tr-1].pop(bisect_left(r_l[tr-1], tc))
                c_l[tc-1].pop(bisect_left(c_l[tc-1], tr))
                d_s.add((tr, tc))
                return st

            return reduce(destroy_wall, valid_targets, (rows_list, cols_list, destroyed))

    # Initial state: (rows, cols, destroyed_set)
    initial_state = (rows, cols, set())
    final_state = reduce(process_query, queries, initial_state)
    
    # Total walls = H * W - size of destroyed set
    print(H * W - len(final_state[2]))

if __name__ == "__main__":
    solve()