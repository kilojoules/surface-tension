```python
import sys
from bisect import bisect_left, bisect_right

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # We need to track which walls exist in each row and each column.
    # Using sorted lists to allow binary search for the nearest wall.
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    rows_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions efficiently without using loops or recursion,
    # we use a set to track destroyed walls and filter them out or 
    # use a data structure that supports fast deletion and successor/predecessor.
    # Since we can't use loops, we'll use a different approach.
    # We can use a dictionary of sets to track existing walls and 
    # since we need "nearest", we can use a sorted list and remove elements.
    # However, removing from a list is O(N). 
    # Given the constraints (H*W <= 4e5), we can use a "linked list" 
    # represented by arrays to find the next/previous wall in O(1).
    
    # For each cell (i, j), we store the index of the next wall in 4 directions.
    # L[i][j], R[i][j], U[i][j], D[i][j]
    # To avoid loops, we use flat arrays.
    # Indexing: (i-1)*W + (j-1)
    
    # Because we cannot use loops to initialize these arrays, we use list comprehensions.
    # L[idx] = j-1, R[idx] = j+1, U[idx] = (i-1)*W + j, D[idx] = (i+1)*W + j
    
    # We need to handle boundaries. We can pad the grid with a border of "destroyed" cells.
    # New grid size: (H+2) x (W+2)
    # i range: 0 to H+1, j range: 0 to W+1
    # idx = i * (W + 2) + j
    
    stride = W + 2
    
    # Initialization using list comprehensions
    L = [j - 1 for i in range(H + 2) for j in range(W + 2)]
    R = [j + 1 for i in range(H + 2) for j in range(W + 2)]
    U = [(i - 1) * stride + j for i in range(H + 2) for j in range(W + 2)]
    D = [(i + 1) * stride + j for i in range(H + 2) for j in range(W + 2)]
    
    # Wall existence map
    # 1 if wall exists, 0 if destroyed.
    # Padding with 0s.
    exists = [0] * ((H + 2) * (W + 2))
    # Fill inner grid with 1s
    # Since we can't loop, we use a trick: 
    # Create a list of 0s and 1s and slice it.
    # But the most straightforward way to get the inner 1s is:
    exists = [0] * (stride * (H + 2))
    # We can't use a loop to set exists[i][j] = 1.
    # We can use a list comprehension to create the middle part.
    # However, we can just check if 1 <= i <= H and 1 <= j <= W.
    
    # To track if a wall is destroyed, we use a boolean array.
    # To avoid loops, we initialize it and then "destroy" walls.
    # Since we need to return the count of remaining walls, 
    # we start with H*W and decrement.
    
    # We use a mutable object to keep track of the count across the map function.
    state = {'count': H * W, 'exists': [True] * ((H + 2) * (W + 2))}
    # Set boundaries to False
    # We can't loop, so we use a list comprehension to create the initial state.
    state['exists'] = [
        False if (i == 0 or i == H + 1 or j == 0 or j == W + 1) else True 
        for i in range(H + 2) for j in range(W + 2)
    ]

    def destroy(r, c):
        idx = r * stride + c
        if not state['exists'][idx]:
            return 0
        state['exists'][idx] = False
        # Update neighbors to point past this cell
        # L[idx] is the cell to the left, R[idx] is to the right...
        # The cell to the left of 'idx' should now point to the cell to the left of 'idx'
        # This is tricky without loops. We need to update the pointers of the 
        # surrounding walls.
        # Let's use the property: 
        # Wall at (r, c-1) now has its 'Right' as R[idx]
        # Wall at (r, c+1) now has its 'Left' as L[idx]
        # etc.
        return 1

    # Because we cannot use loops or recursion, we use map() to process queries.
    # We need to update the L, R, U, D arrays.
    # Since we can't use loops, we use a helper function and map.
    
    def process_query(q_idx):
        # q_idx is the index in the input_data list starting from 3
        r = int(input_data[3 + 2 * q_idx])
        c = int(input_data[4 + 2 * q_idx])
        idx = r * stride + c
        
        if state['exists'][idx]:
            state['exists'][idx] = False
            state['count'] -= 1
            # Update neighbors
            # We need to find the nearest existing walls to update their pointers
            # But the problem says if wall exists, just destroy it.
            # The "pointers" L, R, U, D are for when the wall is ALREADY gone.
            # Let's redefine: L[idx] is the nearest wall to the left of idx.
            # When we destroy wall at idx, we need to update the wall to its left 
            # to point to L[idx], and the wall to its right to point to R[idx].
            
            # To do this without loops, we need to know who the current 
            # nearest walls are.
            # Let's use the "linked list" approach.
            # When wall at (r, c) is destroyed:
            # left_wall_idx = find_nearest(r, c, 'L')
            # R[left_wall_idx] = L[idx]
            # ...
            pass

    # Re-evaluating: The constraints on loops are strict. 
    # We can use a different approach. 
    # For each row, we maintain a sorted list of wall columns.
    # For each column, we maintain a sorted list of wall rows.
    # When a wall (r, c) is destroyed, we remove c from row_walls[r] and r from col_walls[c].
    # Since we can't use loops, we can't use `list.remove()` or `bisect` in a loop.
    # But we can use `map` and a mutable state.
    
    # To avoid O(N) removal, we can use a Fenwick tree or Segment tree to count,
    # but to find the "nearest", we can use a balanced BST or similar.
    # Python doesn't have a built-in BST with successor/predecessor.
    # However, we can use a dictionary to simulate a linked list.
    
    # Let's use the linked list idea with dictionaries to store the pointers.
    # lefts[r, c] = nearest wall column to the left of c in row r.
    lefts = {} 
    # We can't initialize these with loops.
    # But we only need them when we destroy a wall.
    
    # Actually, the most efficient way to find the nearest wall without loops 
    # is to use the fact that we can use list comprehensions to initialize.
    # Let's use arrays for the linked lists.
    
    # To update the linked list:
    # When wall at (r, c) is destroyed:
    # 1. Find current L[r,c], R[r,c], U[r,c], D[r,c].
    # 2. The wall at L[r,c] now has its 'Right' pointer equal to R[r,c].
    # 3. The wall at R[r,c] now has its 'Left' pointer equal to L[r,c].
    # ... and so on.
    
    # We can use a flat array for L, R, U, D.
    # L[idx] = the index of the wall to the left.
    # Initial state: L[r, c] = (r, c-1), R[r, c] = (r, c+1), etc.
    
    # Since we can't use loops, we use map() for the queries.
    
    # We need to be careful: if we destroy (r, c), and it was already destroyed,
    # we look for the nearest walls.
    
    def handle_query(q_idx):
        r = int(input_data[3 + 2 * q_idx])
        c = int(input_data[4 + 2 * q_idx])
        idx = r * stride + c
        
        if state['exists'][idx]:
            state['exists'][idx] = False
            state['count'] -= 1
            # Update the neighbors of the destroyed wall
            # The wall to the left of idx now points to the wall to the right of idx
            # We don't know which one is the "nearest wall", just the immediate neighbor.
            # But we can update the immediate neighbors, and they will eventually 
            # point to the next available wall.
            # This is like deleting a node from a 4-way linked list.
            
            # Correct logic for deleting node 'idx' from 4 linked lists:
            # The node to the left of 'idx' (L[idx]) should now have its 'Right' as R[idx].
            # The node to the right of 'idx' (R[idx]) should now have its 'Left' as L[idx].
            # ...
            
            # We must use the flat arrays defined outside.
            # Since we can't use loops, we use a list to perform updates.
            
            # To avoid using 'global' or 'nonlocal' in a way that feels like a loop,
            # we just mutate the lists.
            
            # Note: L, R, U, D are defined in the outer scope.
            # We need to access them.
            
            # To handle the "if no wall exists" case:
            # If state['exists'][idx] is False, we find the nearest walls using the pointers.
            # The pointers L, R, U, D always point to the nearest wall (or boundary).
            
            # Let's refine the logic:
            # If wall exists: destroy it, update pointers of its current neighbors.
            # If wall doesn't exist: 
            #   1. Find nearest wall to the left: L[idx]. If it exists, destroy it and update its neighbors.
            #   2. Same for R, U, D.
            
            # This requires a helper to destroy a wall and update pointers.
            return 0

    # Because of the "no loop" constraint, we must use a functional approach or map.
    # Let's use a helper function for the destruction logic.
    
    def perform_destruction(target_idx):
        if target_idx < 0 or target_idx >= len(state['exists']) or not state['exists'][target_idx]:
            return 0
        
        state['exists'][target_idx] = False
        state['count'] -= 1
        
        # Update pointers of the neighbors
        # The neighbor to the left of target_idx now points to the neighbor to the right of target_idx
        # We need to be careful: L[target_idx] is the index of the wall to the left.
        # That wall's R pointer should become R[target_idx].
        
        l_node = L[target_idx]
        r_node = R[