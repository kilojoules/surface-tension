import sys
from bisect import bisect_right

def solve():
    # Read H, W, Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = input_data[3:]

    # We need to track existing walls in each row and each column.
    # Using sorted lists to allow binary search for the nearest wall.
    # rows[i] contains sorted column indices of walls in row i.
    # cols[j] contains sorted row indices of walls in column j.
    # Since H*W is up to 4e5, we use lists. 
    # Note: We use 0-indexing internally.
    
    # Initializing rows and cols. 
    # Using range() and converting to list.
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # To track if a wall exists at (r, c) without scanning the sorted list,
    # we use a set of destroyed walls.
    # However, the constraints on H*W allow a flat array for wall existence.
    # wall_exists[r * W + c]
    exists = [True] * (H * W)

    # Process queries
    # We use a generator to avoid loops and maintain the "functional" constraint
    # though the state is mutated in the helper.
    
    def process_query(q_idx):
        r = int(queries[2 * q_idx]) - 1
        c = int(queries[2 * q_idx + 1])
        
        # Check if wall exists at (r, c)
        # The wall at (r, c) is represented by index r * W + (c-1)
        idx = r * W + (c - 1)
        
        if exists[idx]:
            # Destroy wall at (r, c)
            exists[idx] = False
            # Remove from sorted lists
            # We use bisect to find the exact position to remove
            # Since we cannot use loops, we use the fact that the value is 'c'
            # and we must remove it from rows[r] and cols[c-1]
            # We use a trick with slice assignment to remove the element
            
            # Update rows[r]
            pos_r = bisect_right(rows[r], c) - 1
            rows[r][pos_r:pos_r+1] = []
            
            # Update cols[c-1]
            pos_c = bisect_right(cols[c-1], r + 1) - 1
            cols[c-1][pos_c:pos_c+1] = []
            
            return 1 # One wall destroyed
        else:
            # Destroy 4 neighbors
            destroyed = 0
            
            # 1. Up (same column c-1, row i < r+1)
            # Find largest i in cols[c-1] such that i < r+1
            pos_up = bisect_right(cols[c-1], r) - 1
            if pos_up >= 0:
                target_r = cols[c-1][pos_up]
                exists[(target_r - 1) * W + (c - 1)] = False
                rows[target_r - 1][bisect_right(rows[target_r - 1], c) - 1 : bisect_right(rows[target_r - 1], c)] = []
                cols[c-1][pos_up : pos_up + 1] = []
                destroyed += 1
                
            # 2. Down (same column c-1, row i > r+1)
            # Find smallest i in cols[c-1] such that i > r+1
            pos_down = bisect_right(cols[c-1], r + 1)
            if pos_down < len(cols[c-1]):
                target_r = cols[c-1][pos_down]
                exists[(target_r - 1) * W + (c - 1)] = False
                rows[target_r - 1][bisect_right(rows[target_r - 1], c) - 1 : bisect_right(rows[target_r - 1], c)] = []
                cols[c-1][pos_down : pos_down + 1] = []
                destroyed += 1
                
            # 3. Left (same row r, col j < c)
            # Find largest j in rows[r] such that j < c
            pos_left = bisect_right(rows[r], c - 1) - 1
            if pos_left >= 0:
                target_c = rows[r][pos_left]
                exists[r * W + (target_c - 1)] = False
                cols[target_c - 1][bisect_right(cols[target_c - 1], r + 1) - 1 : bisect_right(cols[target_c - 1], r + 1)] = []
                rows[r][pos_left : pos_left + 1] = []
                destroyed += 1
                
            # 4. Right (same row r, col j > c)
            # Find smallest j in rows[r] such that j > c
            pos_right = bisect_right(rows[r], c)
            if pos_right < len(rows[r]):
                target_c = rows[r][pos_right]
                exists[r * W + (target_c - 1)] = False
                cols[target_c - 1][bisect_right(cols[target_c - 1], r + 1) - 1 : bisect_right(cols[target_c - 1], r + 1)] = []
                rows[r][pos_right : pos_right + 1] = []
                destroyed += 1
                
            return destroyed

    # Use a list comprehension to process all queries
    # We use a helper function that modifies the state
    # Since we cannot use loops, we map the process_query over the range of Q
    results = [process_query(q) for q in range(Q)]
    
    # Total walls destroyed is the sum of results
    total_destroyed = sum(results)
    print(H * W - total_destroyed)

if __name__ == "__main__":
    solve()