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
    # To avoid loops, we use a dictionary or list comprehension for initialization.
    
    # rows[i] stores sorted indices of columns that have a wall in row i.
    # cols[j] stores sorted indices of rows that have a wall in column j.
    # Initializing these without loops:
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # To track destroyed walls without loops, we use a set of (r, c) tuples.
    # But we cannot use a loop to process queries. 
    # Wait, the prompt says "Write a complete Python program". 
    # Usually, "no loops" implies using map/reduce/recursion, 
    # but for Q=2e5, recursion will hit limits and map/reduce is still sequential.
    # The core constraint is the logic of the bomb.
    
    # Since I must avoid explicit for/while loops for the query processing:
    # I will use a mutable state object and map().
    
    class State:
        def __init__(self, r_walls, c_walls):
            self.r_walls = r_walls
            self.c_walls = c_walls
            self.destroyed_count = 0

        def process(self, query):
            r, c = query
            # Adjust to 0-indexed for list access
            r_idx, c_idx = r - 1, c - 1
            
            # Check if wall exists at (r, c)
            # We use bisect to check existence in the sorted list
            idx_in_row = bisect_left(self.r_walls[r_idx], c)
            exists = idx_in_row < len(self.r_walls[r_idx]) and self.r_walls[r_idx][idx_in_row] == c
            
            if exists:
                # Destroy wall at (r, c)
                self.r_walls[r_idx].pop(idx_in_row)
                # Find index in col list to pop
                idx_in_col = bisect_left(self.cols_ref[c_idx], r)
                self.cols_ref[c_idx].pop(idx_in_col)
                self.destroyed_count += 1
            else:
                # Destroy 4 neighbors
                # 1. Up (same col, smaller row)
                col_list = self.cols_ref[c_idx]
                idx = bisect_left(col_list, r)
                if idx > 0:
                    target_r = col_list[idx-1]
                    # Remove target_r from col_list and target_c from row_list
                    col_list.pop(idx-1)
                    row_list = self.r_walls[target_r-1]
                    row_idx = bisect_left(row_list, c)
                    row_list.pop(row_idx)
                    self.destroyed_count += 1
                
                # 2. Down (same col, larger row)
                col_list = self.cols_ref[c_idx]
                idx = bisect_left(col_list, r)
                if idx < len(col_list):
                    target_r = col_list[idx]
                    col_list.pop(idx)
                    row_list = self.r_walls[target_r-1]
                    row_idx = bisect_left(row_list, c)
                    row_list.pop(row_idx)
                    self.destroyed_count += 1
                
                # 3. Left (same row, smaller col)
                row_list = self.r_walls[r_idx]
                idx = bisect_left(row_list, c)
                if idx > 0:
                    target_c = row_list[idx-1]
                    row_list.pop(idx-1)
                    col_list = self.cols_ref[target_c-1]
                    col_idx = bisect_left(col_list, r)
                    col_list.pop(col_idx)
                    self.destroyed_count += 1
                
                # 4. Right (same row, larger col)
                row_list = self.r_walls[r_idx]
                idx = bisect_left(row_list, c)
                if idx < len(row_list):
                    target_c = row_list[idx]
                    row_list.pop(idx)
                    col_list = self.cols_ref[target_c-1]
                    col_idx = bisect_left(col_list, r)
                    col_list.pop(col_idx)
                    self.destroyed_count += 1
            return None

    # To avoid the 'no loop' restriction strictly, I'll use a list comprehension 
    # to drive the state changes, but the logic inside process() is the engine.
    # Since I cannot use a loop to initialize cols, I use a list comprehension.
    
    # We need to pass cols into the state. 
    # I'll use a closure or a class attribute.
    
    state = State(rows, cols)
    state.cols_ref = cols
    
    # Use map to process queries without a for loop
    list(map(state.process, queries))
    
    print(H * W - state.destroyed_count)

# Using a wrapper to call solve() without a loop
solve()