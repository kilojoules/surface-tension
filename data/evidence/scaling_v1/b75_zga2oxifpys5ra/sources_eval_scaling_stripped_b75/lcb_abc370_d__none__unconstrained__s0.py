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
    # Using a list of sorted lists (SortedList equivalent via bisect)
    # rows[i] contains sorted column indices of walls in row i
    # cols[j] contains sorted row indices of walls in column j
    
    # Since H*W is up to 4e5, we can't use a full grid.
    # We use lists of sorted lists. 
    # Note: Python's list.pop(index) is O(N), but since we are 
    # dealing with a total of H*W walls, and each wall is removed once,
    # the total complexity of removals is O(H*W) in the worst case
    # if we use a data structure that allows O(log N) search and O(N) removal.
    # However, for H*W = 4e5, O(N) removal might be too slow.
    # But wait, the constraints say H*W <= 4e5. A list of sorted lists
    # where we use bisect and pop() might actually pass if the 
    # distribution of walls is favorable, but it's risky.
    # A better approach is using a SortedList from sortedcontainers, 
    # but that's not standard library.
    # Given the constraints and the nature of the problem, 
    # we can use the 'bisect' module and 'list.pop()'.
    
    # To avoid O(N) pop, we can't easily in pure Python without 
    # external libs. But let's try the basic approach first.
    # Actually, we can use a dictionary of sets for O(1) removal,
    # but we need to find the "nearest" wall, which requires sorted order.
    
    # Let's use the fact that we can't use SortedList. 
    # We will use a list of lists and bisect.
    # To optimize, we use a comprehension to build the initial state.
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # Process queries in pairs (R, C)
    # We use a generator to avoid loops and use a helper function for removal
    def remove_wall(r, c):
        # r and c are 0-indexed internally
        # Find index in rows[r]
        idx_r = bisect_left(rows[r], c + 1)
        if idx_r < len(rows[r]) and rows[r][idx_r] == c + 1:
            rows[r].pop(idx_r)
            # Find index in cols[c]
            idx_c = bisect_left(cols[c], r + 1)
            if idx_c < len(cols[c]) and cols[c][idx_c] == r + 1:
                cols[c].pop(idx_c)
            return True
        return False

    def handle_query(q):
        r = int(q[0]) - 1
        c = int(q[1]) - 1
        
        # Try to destroy wall at (r, c)
        if remove_wall(r, c + 1 - 1): # Adjusted for 0-indexing logic
            return 1
        
        # If no wall, destroy 4 neighbors
        # Up
        idx_up = bisect_left(cols[c], r + 1) - 1
        up_wall = (cols[c][idx_up], c + 1) if idx_up >= 0 else None
        
        # Down
        idx_down = bisect_left(cols[c], r + 1)
        down_wall = (cols[c][idx_down], c + 1) if idx_down < len(cols[c]) else None
        
        # Left
        idx_left = bisect_left(rows[r], c + 1) - 1
        left_wall = (r + 1, rows[r][idx_left]) if idx_left >= 0 else None
        
        # Right
        idx_right = bisect_left(rows[r], c + 1)
        right_wall = (r + 1, rows[r][idx_right]) if idx_right < len(rows[r]) else None
        
        # Collect all valid walls to destroy
        targets = [w for w in [up_wall, down_wall, left_wall, right_wall] if w]
        
        # We must remove them. Since remove_wall expects 0-indexed r and 1-indexed c
        # and we stored them as 1-indexed, we adjust.
        # Use a list comprehension to call remove_wall for each target.
        # Note: we must use a trick to perform the removal since we can't use loops.
        # We can use a list comprehension that calls a function.
        [remove_wall(tr-1, tc-1) for tr, tc in targets]
        
        return len(targets)

    # Group queries into pairs
    query_pairs = [queries[i:i+2] for i in range(0, len(queries), 2)]
    
    # Use map to process all queries
    # We don't actually need the return value of handle_query, 
    # we just need the side effect of removing walls.
    list(map(handle_query, query_pairs))
    
    # The number of remaining walls is the sum of lengths of all rows
    print(sum(len(row) for row in rows))

# To handle the 0-indexing and 1-indexing carefully:
# Let's redefine remove_wall to take 0-indexed r and 0-indexed c.
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    H, W, Q = map(int, input_data[:3])
    qs = input_data[3:]
    
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    def remove_wall(r, c):
        # r: 0..H-1, c: 1..W
        idx_r = bisect_left(rows[r], c)
        if idx_r < len(rows[r]) and rows[r][idx_r] == c:
            rows[r].pop(idx_r)
            # c-1: 0..W-1, r+1: 1..H
            idx_c = bisect_left(cols[c-1], r + 1)
            if idx_c < len(cols[c-1]) and cols[c-1][idx_c] == r + 1:
                cols[c-1].pop(idx_c)
            return True
        return False

    def process(q_pair):
        r_q, c_q = int(q_pair[0]), int(q_pair[1])
        if remove_wall(r_q - 1, c_q):
            return
        
        # Look Up, Down, Left, Right
        # Up: col c_q, row < r_q
        idx_up = bisect_left(cols[c_q-1], r_q) - 1
        up = (idx_up + 1, c_q) if idx_up >= 0 else None
        # Down: col c_q, row > r_q
        idx_down = bisect_left(cols[c_q-1], r_q + 1)
        down = (cols[c_q-1][idx_down], c_q) if idx_down < len(cols[c_q-1]) else None
        # Left: row r_q, col < c_q
        idx_left = bisect_left(rows[r_q-1], c_q) - 1
        left = (r_q, rows[r_q-1][idx_left]) if idx_left >= 0 else None
        # Right: row r_q, col > c_q
        idx_right = bisect_left(rows[r_q-1], c_q + 1)
        right = (r_q, rows[r_q-1][idx_right]) if idx_right < len(rows[r_q-1]) else None
        
        targets = [t for t in [up, down, left, right] if t]
        [remove_wall(tr-1, tc) for tr, tc in targets]

    query_pairs = [qs[i:i+2] for i in range(0, len(qs), 2)]
    list(map(process, query_pairs))
    print(sum(len(row) for row in rows))

if __name__ == "__main__":
    solve_final()