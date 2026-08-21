import sys
from bisect import bisect_left

def solve():
    # Read H, W, Q
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

    # We need to maintain sets of existing walls for every row and every column.
    # Since H*W is up to 4*10^5, we can use sorted lists to simulate sets.
    # rows[i] contains sorted indices of columns that have a wall in row i.
    # cols[j] contains sorted indices of rows that have a wall in column j.
    
    # Using lists and bisect for efficient searching and removal.
    # Note: list.pop(index) is O(N), but given the constraints and the 
    # nature of the problem, we must use a data structure that allows 
    # efficient range queries and updates. 
    # However, Python's list.pop() is surprisingly fast for N=4*10^5 
    # if the number of removals is managed. 
    # To be safe and efficient, we use a SortedList-like approach.
    # Since we cannot import external libraries, we use the fact that 
    # we can maintain sorted lists and use bisect.
    
    from bisect import bisect_left, insort
    
    # Initialize walls: every cell has a wall.
    # To avoid O(H*W) initialization, we only track destroyed walls? 
    # No, the problem asks for remaining walls. 
    # Actually, initializing rows = [list(range(1, W+1)) for _ in range(H)]
    # is O(H*W), which is 4*10^5, acceptable.
    
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # To track if a wall exists at (r, c) efficiently:
    # We can't use a set of all walls because that's H*W.
    # Instead, we check if c is in rows[r-1] using bisect.
    
    def wall_exists(r, c):
        idx = bisect_left(rows[r-1], c)
        return idx < len(rows[r-1]) and rows[r-1][idx] == c

    def destroy(r, c):
        # Remove c from rows[r-1]
        idx_r = bisect_left(rows[r-1], c)
        if idx_r < len(rows[r-1]) and rows[r-1][idx_r] == c:
            rows[r-1].pop(idx_r)
        
        # Remove r from cols[c-1]
        idx_c = bisect_left(cols[c-1], r)
        if idx_c < len(cols[c-1]) and cols[c-1][idx_c] == r:
            cols[c-1].pop(idx_c)

    for r, c in queries:
        if wall_exists(r, c):
            destroy(r, c)
        else:
            # Look Up
            idx = bisect_left(cols[c-1], r)
            if idx > 0:
                destroy(cols[c-1][idx-1], c)
            
            # Look Down
            idx = bisect_left(cols[c-1], r)
            if idx < len(cols[c-1]):
                destroy(cols[c-1][idx], c)
                
            # Look Left
            idx = bisect_left(rows[r-1], c)
            if idx > 0:
                destroy(r, rows[r-1][idx-1])
                
            # Look Right
            idx = bisect_left(rows[r-1], c)
            if idx < len(rows[r-1]):
                destroy(r, rows[r-1][idx])

    # The number of remaining walls is the sum of lengths of all row lists.
    print(sum(len(row) for row in rows))

if __name__ == "__main__":
    solve()