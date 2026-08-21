import sys
from bisect import bisect_left, insort
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

    # We maintain sorted lists of existing wall indices for every row and every column.
    # rows[i] contains columns j where a wall exists at (i, j).
    # cols[j] contains rows i where a wall exists at (i, j).
    # Using lists and bisect allows O(log N) search and O(N) deletion.
    # Given H*W <= 4e5, O(N) deletion in a list is generally acceptable in Python 
    # for these constraints, but we must avoid explicit for/while loops.
    
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # State is (rows, cols, total_walls)
    # We use a helper function to remove a wall from the tracking structures.
    def remove_wall(state, r, c):
        rows, cols, count = state
        # Use list.remove() or pop() via index. 
        # Since we can't use loops, we use the fact that we know the exact indices.
        # We need to find the index of c in rows[r-1] and r in cols[c-1].
        r_idx = bisect_left(rows[r-1], c)
        c_idx = bisect_left(cols[c-1], r)
        
        # Only remove if the wall actually exists
        if r_idx < len(rows[r-1]) and rows[r-1][r_idx] == c:
            rows[r-1].pop(r_idx)
            cols[c-1].pop(c_idx)
            return (rows, cols, count - 1)
        return state

    def process_query(state, query):
        rows, cols, count = state
        r, c = query
        
        # Check if wall exists at (r, c)
        r_idx = bisect_left(rows[r-1], c)
        exists = r_idx < len(rows[r-1]) and rows[r-1][r_idx] == c
        
        if exists:
            return remove_wall(state, r, c)
        
        # If no wall, find nearest walls in 4 directions
        # Up: column c, largest i < r
        # Down: column c, smallest i > r
        # Left: row r, largest j < c
        # Right: row r, smallest j > c
        
        # Column search (Up/Down)
        c_list = cols[c-1]
        c_pos = bisect_left(c_list, r)
        
        # Row search (Left/Right)
        r_list = rows[r-1]
        r_pos = bisect_left(r_list, c)
        
        # Identify targets
        targets = []
        # Up
        if c_pos > 0: targets.append((c_list[c_pos-1], c))
        # Down
        if c_pos < len(c_list): targets.append((c_list[c_pos], c))
        # Left
        if r_pos > 0: targets.append((r, r_list[r_pos-1])) # Wait, r_list contains columns
        # Correcting:
        # Left: row r, column r_list[r_pos-1]
        # Right: row r, column r_list[r_pos]
        
        # Let's redefine targets clearly
        up = (c_list[c_pos-1], c) if c_pos > 0 else None
        down = (c_list[c_pos], c) if c_pos < len(c_list) else None
        left = (r, r_list[r_pos-1]) if r_pos > 0 else None
        right = (r, r_list[r_pos]) if r_pos < len(r_list) else None
        
        # Apply removals using reduce to avoid loops
        return reduce(
            lambda s, t: remove_wall(s, t[0], t[1]) if t else s,
            [up, down, left, right],
            state
        )

    # Initial state
    start_state = (initial_rows, initial_cols, H * W)
    
    # Process all queries
    final_state = reduce(process_query, queries, start_state)
    
    # The result is the third element of the state tuple
    print(final_state[2])

if __name__ == "__main__":
    solve()