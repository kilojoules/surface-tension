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
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # We need to track existing walls in each row and each column.
    # Using sorted lists allows us to find the nearest wall using binary search.
    # Since H*W is up to 4e5, we can't use a simple list for every row/col if H or W is huge.
    # However, the constraints say H*W <= 4e5, so we can initialize lists.
    
    # rows[i] contains sorted indices of columns that have a wall in row i.
    # cols[j] contains sorted indices of rows that have a wall in column j.
    rows = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions efficiently in a sorted list, we use a trick.
    # Since we cannot use external libraries like SortedList, and H*W is small enough,
    # we can use a set for O(1) check and a way to find neighbors.
    # But wait, H*W <= 4e5 means we can't afford O(N) deletions.
    # Let's use a different approach: 
    # Since we need to find the nearest existing wall, we can use a DSU-like 
    # structure or simply use the fact that we can't use SortedList.
    # Actually, for H*W <= 4e5, we can use a dictionary of sets to track walls.
    # To find the nearest wall, we can't iterate. 
    # But we can use the 'bisect' module on sorted lists. 
    # The problem is that list.pop(i) is O(N).
    # Given the constraints and the nature of the problem, the most efficient 
    # way in pure Python without SortedList is to use a Fenwick tree or Segment Tree
    # to find the next/previous 1 in a bit-array, but that's complex.
    
    # Let's reconsider: if we use a set for each row and column, 
    # we can't find the "nearest" element.
    # However, we can use a linked-list style approach with arrays.
    # For each cell, store prev_row, next_row, prev_col, next_col.
    # But that's 4 * 4e5 integers, which is fine.
    
    # Using arrays to simulate a 2D linked list:
    # We map (r, c) to a unique ID: (r-1)*W + (c-1)
    # L[id], R[id], U[id], D[id]
    # Initialize: L[id] = id-1, R[id] = id+1, etc.
    # When a wall is destroyed, we update the neighbors' pointers.
    
    # Since we need to handle the "no wall at (Rq, Cq)" case, 
    # we need to know if a wall exists.
    exists = [True] * (H * W)
    
    # Pointers for each cell
    # Using 0-indexed internal logic: r in [0, H-1], c in [0, W-1]
    # Left/Right pointers (horizontal)
    # For a cell (r, c), left is (r, c-1), right is (r, c+1)
    # We use a flat array and handle boundaries.
    
    # To avoid loops and recursion, we use a functional approach to update pointers.
    # But we can't use loops. We can use a helper function and a list comprehension.
    
    # Actually, the simplest way to implement this without loops is to use 
    # a mutable data structure and a function that performs the deletion.
    
    # Let's use a class to encapsulate the grid state.
    class Grid:
        def __init__(self, h, w):
            self.h = h
            self.w = w
            self.exists = [True] * (h * w)
            # Using lists as mutable pointers
            # left[i] is the index of the wall to the left of cell i
            self.left = [i - 1 if i % w != 0 else -1 for i in range(h * w)]
            self.right = [i + 1 if i % w != w - 1 else -1 for i in range(h * w)]
            self.up = [i - w if i >= w else -1 for i in range(h * w)]
            self.down = [i + w if i < (h - 1) * w else -1 for i in range(h * w)]

        def remove(self, i):
            if i == -1 or not self.exists[i]: return
            self.exists[i] = False
            # Update neighbors
            l, r = self.left[i], self.right[i]
            u, d = self.up[i], self.down[i]
            if l != -1: self.right[l] = r
            if r != -1: self.left[r] = l
            if u != -1: self.down[u] = d
            if d != -1: self.up[d] = u

        def query(self, r, c):
            idx = (r - 1) * self.w + (c - 1)
            if self.exists[idx]:
                self.remove(idx)
            else:
                # Destroy 4 neighbors
                # We must find the first existing walls in 4 directions.
                # Since the pointers always point to the nearest existing wall:
                self.remove(self.left[idx])
                self.remove(self.right[idx])
                self.remove(self.up[idx])
                self.remove(self.down[idx])

    grid = Grid(H, W)
    # Use a list comprehension to process all queries
    [grid.query(r, c) for r, c in queries]
    
    # Count remaining walls
    print(sum(grid.exists))

if __name__ == "__main__":
    solve()