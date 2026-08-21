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
    
    # Queries as a list of (R, C) tuples
    queries = [ (int(input_data[i]), int(input_data[i+1])) 
                for i in range(3, len(input_data), 2) ]
    
    # Grid represented as a 1D list: index = (r-1)*W + (c-1)
    # 1 means wall exists, 0 means destroyed
    initial_grid = [1] * (H * W)
    
    # Helper to get 1D index
    # idx = lambda r, c: (r - 1) * W + (c - 1)
    
    def process_query(grid, query):
        r, c = query
        idx = (r - 1) * W + (c - 1)
        
        # If wall exists at (r, c), destroy it and return
        if grid[idx] == 1:
            # Use slice assignment to modify the list in place (simulated)
            # Since we need to return the grid for reduce, we modify and return
            grid[idx] = 0
            return grid
        
        # If no wall, find first walls in 4 directions
        # We use list comprehensions and next() to find the first '1'
        
        # Up: r decreases
        up_idx = next([ (i - 1) * W + (c - 1) 
                        for i in range(r - 1, 0, -1) 
                        if grid[(i - 1) * W + (c - 1)] == 1 ], None)
        
        # Down: r increases
        down_idx = next([ (i - 1) * W + (c - 1) 
                          for i in range(r + 1, H + 1) 
                          if grid[(i - 1) * W + (c - 1)] == 1 ], None)
        
        # Left: c decreases
        left_idx = next([ (r - 1) * W + (j - 1) 
                          for j in range(c - 1, 0, -1) 
                          if grid[(r - 1) * W + (j - 1)] == 1 ], None)
        
        # Right: c increases
        right_idx = next([ (r - 1) * W + (j - 1) 
                           for j in range(c + 1, W + 1) 
                           if grid[(r - 1) * W + (j - 1)] == 1 ], None)
        
        # Apply destructions
        # We use a list of indices to destroy and a loop-free way to update
        # Since we can't use for loops, we use a conditional update
        # Note: grid is mutable, so we can update it.
        # To avoid 'for', we can use a map or a list comprehension that modifies
        [ (grid.__setitem__(i, 0)) for i in [up_idx, down_idx, left_idx, right_idx] if i is not None ]
        
        return grid

    # Use reduce to process all queries sequentially
    final_grid = reduce(process_query, queries, initial_grid)
    
    # Count remaining walls using sum()
    print(sum(final_grid))

if __name__ == "__main__":
    solve()