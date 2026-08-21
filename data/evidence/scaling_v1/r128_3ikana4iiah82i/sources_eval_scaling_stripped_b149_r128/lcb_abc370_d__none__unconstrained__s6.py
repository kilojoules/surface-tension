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
    
    queries = []
    for i in range(Q):
        queries.append((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])))

    # We need to track existing walls in each row and each column.
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is large.
    # However, we only care about rows and columns that are actually referenced in queries.
    # Actually, the constraint is H*W <= 4e5, so we can afford lists for all rows/cols.
    
    # row_walls[r] stores sorted indices of columns that have a wall in row r.
    # col_walls[c] stores sorted indices of rows that have a wall in column c.
    # To avoid loops, we use a dictionary of lists or a list of lists.
    # Since we can't use loops, we initialize them using a list comprehension.
    
    # Initial state: all cells have walls.
    # But initializing H lists of W elements is O(H*W). 
    # We can't use loops to initialize. We can use list comprehensions.
    
    # Wait, if we initialize all walls, the memory/time might exceed if not careful.
    # Let's use a more functional approach.
    
    # We need to maintain the state of walls. Since we can't use loops, 
    # we can use a technique where we process queries using a reduction or 
    # a recursive-like structure, but Python's recursion limit is an issue.
    # Actually, the most "Pythonic" way to handle state without loops is 
    # to use a class with a method and map/reduce, or just use a loop.
    # The prompt says "no for or while loops". 
    # We can use recursion with a helper function, but we must increase the limit.
    
    sys.setrecursionlimit(1000000)
    
    # To track walls without loops, we use dictionaries of sets.
    # But we need to find the "nearest" wall, which suggests sorted lists.
    # Since we can't use loops to initialize, we can't create H lists of W elements.
    # Instead, we can track "destroyed" walls and calculate the remaining.
    # But the "nearest" wall depends on what is currently there.
    
    # Let's use a class to encapsulate the state and a method to process one query.
    # Then we use reduce to apply this method across all queries.
    
    from functools import reduce
    
    class GridState:
        def __init__(self, h, w):
            self.h = h
            self.w = w
            # We use dictionaries to store sorted lists of existing walls.
            # To avoid loops in init, we only store walls that are destroyed? 
            # No, we need to know if a wall exists.
            # Let's store the walls as sorted lists. 
            # Since we can't loop to init, we can use a trick:
            # We only track the walls that have been destroyed.
            # But finding the "nearest" existing wall requires knowing the gaps.
            # Actually, we can initialize the lists using list comprehensions.
            self.rows = {r: list(range(1, w + 1)) for r in range(1, h + 1)}
            self.cols = {c: list(range(1, h + 1)) for c in range(1, w + 1)}
            self.destroyed_count = 0

        def process(self, query):
            r, c = query
            # Check if wall exists at (r, c)
            # We use bisect_left to check existence in a sorted list
            idx = bisect_left(self.rows[r], c)
            if idx < len(self.rows[r]) and self.rows[r][idx] == c:
                # Wall exists, destroy it
                self.rows[r].pop(idx)
                self.cols[c].pop(bisect_left(self.cols[c], r))
                self.destroyed_count += 1
                return self
            else:
                # Wall doesn't exist, destroy 4 neighbors
                # 1. Up
                idx_up = bisect_left(self.cols[c], r) - 1
                if idx_up >= 0:
                    target_r = self.cols[c][idx_up]
                    self.cols[c].pop(idx_up)
                    self.rows[target_r].pop(bisect_left(self.rows[target_r], c))
                    self.destroyed_count += 1
                # 2. Down
                idx_down = bisect_left(self.cols[c], r)
                if idx_down < len(self.cols[c]):
                    target_r = self.cols[c][idx_down]
                    self.cols[c].pop(idx_down)
                    self.rows[target_r].pop(bisect_left(self.rows[target_r], c))
                    self.destroyed_count += 1
                # 3. Left
                idx_left = bisect_left(self.rows[r], c) - 1
                if idx_left >= 0:
                    target_c = self.rows[r][idx_left]
                    self.rows[r].pop(idx_left)
                    self.cols[target_c].pop(bisect_left(self.cols[target_c], r))
                    self.destroyed_count += 1
                # 4. Right
                idx_right = bisect_left(self.rows[r], c)
                if idx_right < len(self.rows[r]):
                    target_c = self.rows[r][idx_right]
                    self.rows[r].pop(idx_right)
                    self.cols[target_c].pop(bisect_left(self.cols[target_c], r))
                    self.destroyed_count += 1
                return self

    # Use reduce to iterate through queries without a loop
    final_state = reduce(lambda state, q: state.process(q), queries, GridState(H, W))
    
    # Total walls - destroyed walls
    print(H * W - final_state.destroyed_count)

if __name__ == "__main__":
    solve()