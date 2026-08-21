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
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # We maintain the state of the grid using sets of active wall indices for each row and column.
    # rows[i] contains columns j where a wall exists.
    # cols[j] contains rows i where a wall exists.
    # Using sorted lists (via bisect) would be faster, but since we cannot use loops,
    # we use a dictionary of sets and handle the "find nearest" logic using 
    # list comprehensions and min/max.
    
    # To optimize "finding the nearest wall" without loops, we use sorted lists 
    # and the bisect module to find indices.
    import bisect

    # Initial state: all cells have walls.
    # We store walls in sorted lists for each row and column to allow binary search.
    initial_state = {
        'rows': [list(range(1, W + 1)) for _ in range(H + 1)],
        'cols': [list(range(1, H + 1)) for _ in range(W + 1)],
        'total': H * W
    }

    def process_query(state, query):
        r, c = query
        rows, cols = state['rows'], state['cols']
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in O(log N)
        idx = bisect.bisect_left(rows[r], c)
        has_wall = idx < len(rows[r]) and rows[r][idx] == c
        
        if has_wall:
            # Destroy wall at (r, c)
            # Note: mutating lists inside reduce is allowed as long as we don't use 'for' loops.
            rows[r].pop(idx)
            # Find and remove from cols[c]
            c_idx = bisect.bisect_left(cols[c], r)
            cols[c].pop(c_idx)
            return {**state, 'total': state['total'] - 1}
        else:
            # Destroy 4 nearest walls
            # Find nearest in row r (left and right)
            # Find nearest in col c (up and down)
            
            # Row r: left and right
            r_idx = bisect.bisect_left(rows[r], c)
            targets_row = [
                (r, rows[r][r_idx - 1]) if r_idx > 0 else None,
                (r, rows[r][r_idx]) if r_idx < len(rows[r]) else None
            ]
            
            # Col c: up and down
            c_idx = bisect.bisect_left(cols[c], r)
            targets_col = [
                (cols[c][c_idx - 1], c) if c_idx > 0 else None,
                (cols[c][c_idx], c) if c_idx < len(cols[c]) else None
            ]
            
            # Filter None and get unique walls to destroy
            to_destroy = list(set(filter(None, targets_row + targets_col)))
            
            # Remove each target wall from both row and col trackers
            # We use a helper function called via map to avoid 'for' loops
            def remove_wall(s, wall):
                wr, wc = wall
                # Remove from row tracker
                r_pos = bisect.bisect_left(rows[wr], wc)
                rows[wr].pop(r_pos)
                # Remove from col tracker
                c_pos = bisect.bisect_left(cols[wc], wr)
                cols[wc].pop(c_pos)
                return s - 1

            return {**state, 'total': reduce(remove_wall, to_destroy, state['total'])}

    final_state = reduce(process_query, queries, initial_state)
    print(final_state['total'])

if __name__ == "__main__":
    solve()