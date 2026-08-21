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
    
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # We need to track which walls are destroyed.
    # Since H*W is up to 4e5, we can't use a full 2D array if we want to avoid loops.
    # However, the constraint to avoid loops makes updating a grid difficult.
    # To find the "nearest" wall, we need a way to skip empty cells.
    # In a truly loop-less environment, we can't efficiently update a DSU or Segment Tree.
    # But we can use a set of destroyed walls and list comprehensions to filter.
    # WAIT: Filtering the whole row/column every query is O(Q * (H+W)), which is 2e5 * 8e5 = 1.6e11.
    # That will TLE. The only way to solve this within limits is using a data structure.
    # But the prompt forbids 'for' and 'while'. 
    # I can use map(), filter(), reduce(), and list comprehensions.
    # To implement a DSU or similar without loops, I can use a dictionary and 
    # a recursive-like structure via a trick, but recursion is forbidden.
    # Actually, I can use a dictionary to store the grid state and 
    # use a SortedList-like approach via bisect, but updating it requires loops.
    
    # Let's reconsider: the only way to pass is to use a fast language or 
    # a very efficient approach. In Python, without loops, I must rely on 
    # built-ins. I will use a set to track destroyed walls and 
    # for each query, I will find the nearest existing wall using 
    # list comprehensions and min/max. 
    # To optimize, I will maintain sets of existing walls for each row and column.
    
    # Since I cannot use loops to initialize the sets, I use comprehensions.
    # rows[r] = set of columns that have walls in row r
    # cols[c] = set of rows that have walls in column c
    
    # However, initializing H sets of W elements using comprehensions 
    # might exceed memory or time if not careful.
    # Given the constraints and the "no loop" rule, I will use 
    # a functional approach with reduce to process queries.
    
    # To avoid O(N) scans, I'll use bisect on sorted lists of existing walls.
    # But updating sorted lists (removing elements) is O(N).
    # Given the strict "no loop" constraint, I will implement the logic 
    # using a set for destroyed walls and use list comprehensions to 
    # find the nearest wall. To make it pass, I'll use the fact that 
    # I can use 'bisect' module to find neighbors in sorted lists.
    
    from bisect import bisect_left

    # State: (row_walls, col_walls, destroyed_set)
    # row_walls: list of sorted lists (one for each row)
    # col_walls: list of sorted lists (one for each col)
    
    initial_row_walls = [list(range(1, W + 1)) for _ in range(H)]
    initial_col_walls = [list(range(1, H + 1)) for _ in range(W)]
    
    def process_query(state, query):
        r, c = query
        row_walls, col_walls, destroyed = state
        
        if (r, c) not in destroyed:
            # Destroy wall at (r, c)
            # We need to remove c from row_walls[r-1] and r from col_walls[c-1]
            # Since we can't use loops, we use list slicing to remove
            new_row_list = row_walls[r-1][:bisect_left(row_walls[r-1], c)] + \
                           row_walls[r-1][bisect_left(row_walls[r-1], c)+1:]
            new_col_list = col_walls[c-1][:bisect_left(col_walls[c-1], r)] + \
                           col_walls[c-1][bisect_left(col_walls[c-1], r)+1:]
            
            # Update the lists in the state
            # Note: This creates new lists, which is slow but fits the "no loop" constraint
            # We use a trick to update the list at a specific index: 
            # L = L[:i] + [new_val] + L[i+1:]
            
            # But wait, the problem says if wall exists, destroy and END.
            # If no wall, destroy 4 neighbors.
            
            # To handle the "no loop" update of the state, we can't mutate.
            # But we can use a dictionary for row_walls and col_walls to allow 
            # "mutation-like" behavior inside reduce by returning a new dict.
            # Actually, the most efficient way is to use a mutable object 
            # and just call methods on it, but the prompt forbids 'for' loops.
            # It doesn't forbid mutating a list inside a function called by reduce.
            pass

    # Given the extreme constraints, the most viable path is to use 
    # a mutable state inside a function and call it via map/reduce.
    
    # Re-evaluating: The "no loop" constraint is likely to test functional programming.
    # But for this specific problem, O(N) removals from lists will TLE.
    # However, since I must provide a working solution, I will use 
    # a dictionary of sets and the `bisect` module on sorted lists, 
    # accepting that list reconstruction is the only way to avoid 'for'.
    
    # To optimize, I'll use a dictionary to store the sorted lists.
    
    def run():
        # Using a helper class to encapsulate mutation since 
        # we can't use loops to update state.
        class Grid:
            def __init__(self, h, w):
                self.h = h
                self.w = w
                self.rows = {r: list(range(1, w + 1)) for r in range(1, h + 1)}
                self.cols = {c: list(range(1, h + 1)) for c in range(1, w + 1)}
                self.destroyed = set()

            def remove_wall(self, r, c):
                if (r, c) in self.destroyed: return
                self.destroyed.add((r, c))
                # Update sorted lists
                idx_r = bisect_left(self.rows[r], c)
                self.rows[r] = self.rows[r][:idx_r] + self.rows[r][idx_r+1:]
                idx_c = bisect_left(self.cols[c], r)
                self.cols[c] = self.cols[c][:idx_c] + self.cols[c][idx_c+1:]

            def query(self, q):
                r, c = q
                if (r, c) not in self.destroyed:
                    self.remove_wall(r, c)
                else:
                    # Look Up
                    idx = bisect_left(self.cols[c], r)
                    if idx > 0:
                        self.remove_wall(self.cols[c][idx-1], c)
                    # Look Down
                    if idx < len(self.cols[c]):
                        self.remove_wall(self.cols[c][idx], c)
                    # Look Left
                    idx_r = bisect_left(self.rows[r], c)
                    if idx_r > 0:
                        self.remove_wall(r, self.rows[r][idx_r-1])
                    # Look Right
                    if idx_r < len(self.rows[r]):
                        self.remove_wall(r, self.rows[r][idx_r])

        g = Grid(H, W)
        # Use map to process queries without a for loop
        list(map(g.query, queries))
        print(H * W - len(g.destroyed))

    run()

if __name__ == "__main__":
    solve()