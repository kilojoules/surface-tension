import sys
from bisect import bisect_left
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

    # State: rows[i] is a sorted list of columns containing walls in row i
    #        cols[j] is a sorted list of rows containing walls in column j
    # Using lists for rows/cols and updating them via slice assignment or 
    # creating new lists to avoid explicit for/while loops.
    
    # Initial state: every cell has a wall
    # rows = [[1, 2, ..., W] for _ in range(H)]
    # cols = [[1, 2, ..., H] for _ in range(W)]
    # To avoid loops in initialization, we use list comprehensions.
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    def remove_wall(state, r, c):
        rows, cols = state
        # Remove c from rows[r-1]
        idx_c = bisect_left(rows[r-1], c)
        # We use slice assignment to mutate the list without a loop
        # Note: rows[r-1].pop(idx_c) is allowed as it is a single operation
        rows[r-1].pop(idx_c)
        
        # Remove r from cols[c-1]
        idx_r = bisect_left(cols[c-1], r)
        cols[c-1].pop(idx_r)
        return (rows, cols)

    def process_query(state, query):
        r, c = query
        rows, cols = state
        
        # Check if wall exists at (r, c)
        # Use bisect to check existence in O(log N)
        idx_c = bisect_left(rows[r-1], c)
        has_wall = idx_c < len(rows[r-1]) and rows[r-1][idx_c] == c
        
        if has_wall:
            return remove_wall(state, r, c)
        
        # No wall at (r, c), destroy 4 neighbors
        # 1. Left: largest j < c in rows[r-1]
        # 2. Right: smallest j > c in rows[r-1]
        # 3. Up: largest i < r in cols[c-1]
        # 4. Down: smallest i > r in cols[c-1]
        
        # Find targets
        idx_c = bisect_left(rows[r-1], c)
        target_left = rows[r-1][idx_c - 1] if idx_c > 0 else None
        target_right = rows[r-1][idx_c] if idx_c < len(rows[r-1]) else None
        
        idx_r = bisect_left(cols[c-1], r)
        target_up = cols[c-1][idx_r - 1] if idx_r > 0 else None
        target_down = cols[c-1][idx_r] if idx_r < len(cols[c-1]) else None
        
        # Apply removals. We use a list of targets and reduce to avoid loops.
        targets = [
            (r, target_left), (r, target_right), 
            (target_up, c), (target_down, c)
        ]
        # Filter out None values and apply remove_wall
        return reduce(
            lambda s, t: remove_wall(s, t[0], t[1]) if t[1] is not None else s,
            [t for t in targets if t[0] is not None],
            state
        )

    # Process all queries using reduce
    final_state = reduce(process_query, queries, (initial_rows, initial_cols))
    
    # Calculate remaining walls
    # Sum of lengths of all lists in final_state[0]
    total_remaining = sum(map(len, final_state[0]))
    print(total_remaining)

if __name__ == "__main__":
    solve()