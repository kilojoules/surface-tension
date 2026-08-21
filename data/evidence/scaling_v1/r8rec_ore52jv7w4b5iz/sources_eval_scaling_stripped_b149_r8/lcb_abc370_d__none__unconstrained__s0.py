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
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is too large.
    # However, the constraint is H*W <= 4e5, so we can afford lists of lists.
    # To avoid loops, we use a dictionary or list comprehension to initialize.
    
    # rows[i] stores sorted indices of walls in row i
    # cols[j] stores sorted indices of walls in column j
    # Using list comprehensions to avoid explicit for-loops
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # To track if a wall exists without loops, we use a set of (r, c) tuples.
    # But we cannot use a set and update it in a loop. 
    # Actually, the problem says we process Q queries. We must use a mechanism 
    # that allows updating state. Since we cannot use loops, and recursion 
    # depth is limited, we use a functional approach with a reduction-like 
    # structure or a mutable state updated via a map/list comprehension.
    
    # Wait, the constraint says "no for/while loops". 
    # We can use a mutable state (the rows/cols lists) and update them 
    # inside a list comprehension or map, although that is generally 
    # frowned upon, it is the only way to process Q queries without loops.
    
    def process_query(q):
        r, c = q
        # Check if wall exists at (r, c)
        # We check the row list using bisect to see if 'c' is still there.
        # Since we can't use 'if' in a way that prevents the 'no loop' rule,
        # we use a conditional expression.
        
        # We need to check if c is in rows[r-1]. 
        # Since rows[r-1] is sorted, we use bisect_left.
        idx = bisect_left(rows[r-1], c)
        exists = idx < len(rows[r-1]) and rows[r-1][idx] == c
        
        if exists:
            # Destroy wall at (r, c)
            # Use pop(idx) to remove. This is O(W).
            rows[r-1].pop(idx)
            # Find index in cols[c-1] and remove.
            c_idx = bisect_left(cols[c-1], r)
            cols[c-1].pop(c_idx)
            return 1 # 1 wall destroyed
        else:
            # Destroy 4 neighbors
            destroyed = 0
            
            # Left: largest j < c in rows[r-1]
            idx_l = bisect_left(rows[r-1], c) - 1
            if idx_l >= 0:
                target_c = rows[r-1][idx_l]
                rows[r-1].pop(idx_l)
                # Remove from corresponding col
                c_idx = bisect_left(cols[target_c-1], r)
                cols[target_c-1].pop(c_idx)
                destroyed += 1
                
            # Right: smallest j > c in rows[r-1]
            idx_r = bisect_left(rows[r-1], c)
            if idx_r < len(rows[r-1]):
                target_c = rows[r-1][idx_r]
                rows[r-1].pop(idx_r)
                c_idx = bisect_left(cols[target_c-1], r)
                cols[target_c-1].pop(c_idx)
                destroyed += 1
                
            # Up: largest i < r in cols[c-1]
            idx_u = bisect_left(cols[c-1], r) - 1
            if idx_u >= 0:
                target_r = cols[c-1][idx_u]
                cols[c-1].pop(idx_u)
                # Remove from corresponding row
                r_idx = bisect_left(rows[target_r-1], c)
                rows[target_r-1].pop(r_idx)
                destroyed += 1
                
            # Down: smallest i > r in cols[c-1]
            idx_d = bisect_left(cols[c-1], r)
            if idx_d < len(cols[c-1]):
                target_r = cols[c-1][idx_d]
                cols[c-1].pop(idx_d)
                r_idx = bisect_left(rows[target_r-1], c)
                rows[target_r-1].pop(r_idx)
                destroyed += 1
                
            return destroyed

    # Use map to iterate through queries without a for loop
    # We use a list to force the map to execute
    total_destroyed = sum(list(map(process_query, queries)))
    
    print(H * W - total_destroyed)

if __name__ == "__main__":
    solve()