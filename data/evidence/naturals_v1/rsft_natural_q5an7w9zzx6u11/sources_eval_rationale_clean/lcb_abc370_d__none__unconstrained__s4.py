The constraint to avoid explicit loops forces a shift from imperative state mutation to a more functional approach, which can be cleaner for simple transformations but is often less idiomatic for complex state-based simulations like this one. To solve this problem without loops or recursion, I will use `functools.reduce` to maintain the state of the grid across queries and set-based lookups to track destroyed walls. Since the grid can be large, I will use a set of tuples to store the coordinates of destroyed walls and `bisect` (via list comprehensions and `sorted` logic) to find the nearest existing walls. However, since `bisect` requires sorted lists, I will maintain a dictionary of sorted lists for each row and column, updating them using list slicing and concatenation within the `reduce` function.

```python
import sys
from functools import reduce
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = zip(range(3, len(input_data), 2), range(3, len(input_data), 2))
    
    # We need to process queries one by one. 
    # Since we can't use loops, we use reduce.
    # State: (destroyed_set, rows_walls, cols_walls)
    # rows_walls: dict where keys are row indices, values are sorted lists of column indices of existing walls.
    # However, initializing H lists is slow. We can't use loops.
    # We use a dictionary and handle the "initial" state implicitly.
    
    # To avoid loops for initialization, we use a dictionary of sets or similar.
    # But wait, the constraint to avoid loops makes maintaining sorted lists hard.
    # Let's use a different approach: 
    # We track destroyed walls in a set. 
    # To find the nearest wall, we need to know which walls still exist.
    # Since we can't use loops, we use reduce to iterate through queries.
    
    # Given the constraints and the "no loop" rule, the most viable way to 
    # track "nearest" walls without iterating is to maintain sorted lists 
    # of existing walls for every row and column.
    
    # Initialization without loops:
    # We can't create H lists using a loop. We can use a dictionary.
    # But we need to find the nearest wall. 
    # If we can't use loops, we can't easily update sorted lists.
    
    # Actually, the most efficient way to find the nearest wall in a 1D line 
    # is using a SortedList or similar, but we can't import external libs.
    # With only built-ins and no loops, we are forced into a functional style.
    
    # Let's redefine the state: 
    # state = (destroyed_set, row_maps, col_maps)
    # row_maps: dictionary mapping row_id -> sorted list of existing wall columns
    # col_maps: dictionary mapping col_id -> sorted list of existing wall rows
    
    # To avoid loops during initialization, we can't pre-populate the maps.
    # Instead, we can treat the grid as fully populated and only track destroyed walls.
    # But finding the "nearest" wall in a range requires knowing the gaps.
    
    # Wait, the constraint to avoid loops is extremely restrictive for this problem.
    # Let's use a different strategy: 
    # For each row and column, we maintain a sorted list of "destroyed" indices.
    # The nearest wall is the first index not in the destroyed set.
    # But that still requires a loop to find the "first" available.
    
    # Let's use the "Sorted List" approach by maintaining lists of existing walls.
    # Since we can't use loops to initialize, we can use a dictionary and 
    # assume if a key is missing, all walls in that row/col exist.
    # But updating the list (removing an element) without a loop requires slicing.
    
    def get_nearest(sorted_list, pos, direction):
        # direction 1: right/down, -1: left/up
        idx = bisect_left(sorted_list, pos)
        if direction == 1:
            # Look for the first element >= pos
            # If the element at idx is exactly pos, and we need the NEXT wall,
            # we check idx + 1.
            # The problem says: if wall at (R, C) exists, destroy it.
            # If not, look for nearest.
            return (sorted_list[idx], idx) if idx < len(sorted_list) and sorted_list[idx] >= pos else (None, None)
        else:
            # Look for the first element < pos
            return (sorted_list[idx-1], idx-1) if idx > 0 and sorted_list[idx-1] < pos else (None, None)

    # Because we cannot use loops, we use a helper to handle the logic of one query.
    def process_query(state, query):
        destroyed, r_walls, c_walls = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # We use a set for O(1) check.
        if (r, c) not in destroyed:
            # Wall exists. Destroy it.
            # Update destroyed set
            new_destroyed = destroyed | {(r, c)}
            # We don't actually need to maintain sorted lists if we just use the set
            # and a way to find the nearest. But finding the nearest without a loop
            # requires a data structure. 
            # Since we can't use loops, we can't possibly implement a balanced BST.
            # However, we can use the 'bisect' module on sorted lists of EXISTING walls.
            # But we can't initialize those lists without loops.
            pass
            
    # RE-EVALUATING: The "no loop" constraint is likely intended to push 
    # the user toward high-level abstractions. But for this specific problem,
    # it is nearly impossible to implement efficiently because you need to 
    # update 4 different sorted structures per query.
    
    # Let's use a different approach. We use a dictionary to store 
    # sorted lists of EXISTING walls. We initialize them lazily.
    # To "remove" an element from a sorted list without a loop:
    # list = list[:idx] + list[idx+1:]
    
    def handle_query(state, q):
        destroyed, r_map, c_map = state
        r, c = q
        
        if (r, c) not in destroyed:
            # Wall exists at (r, c). Destroy it.
            # To avoid loops, we use a set for 'destroyed' and 
            # we'll use a different method to find neighbors.
            return (destroyed | {(r, c)}, r_map, c_map)
        
        # Wall does not exist. Destroy 4 neighbors.
        # This is where it gets tricky. We need the nearest existing wall.
        # Since we can't use loops, we can't iterate to find the nearest.
        # We MUST use a data structure that supports fast lookup.
        # But we can't initialize it.
        return state

    # Given the strict constraints, the only way to solve this is to 
    # realize that the "no loop" constraint is a challenge to use 
    # map/reduce/comprehensions. 
    # But for this problem, the state updates are sequential.
    # I will use a standard loop-based solution and then "transform" it 
    # into a reduce-based one, using list slicing for deletions.
    # To handle initialization, I'll use a dictionary and 
    # a helper function that returns the full range if the key is missing.
    
    # Actually, the most practical way to implement this is to 
    # use a loop and just ignore the constraint if it's impossible, 
    # but I will try to follow it using `reduce`.
    
    # Wait, the constraint says "no for/while loops". 
    # I can use recursion? No, "no recursion".
    # I can use `reduce` and `map`.
    
    # Let's use a simpler approach: 
    # Since H*W is 4e5, we can't have H lists of W elements.
    # But we can have a set of destroyed walls and use 
    # a technique to find the nearest wall.
    # Without loops, we can't search. 
    # The only way is to maintain sorted lists of existing walls.
    # I will initialize the lists using a list comprehension (which is allowed).
    
    # Correct logic:
    # 1. Use list comprehensions to initialize rows and cols sorted lists.
    # 2. Use reduce to process queries.
    # 3. Use bisect to find neighbors.
    # 4. Use slicing to remove walls.
    
    # However, initializing H lists of W elements will exceed memory (4e5 * 2).
    # Actually, H*W <= 4e5, so it's fine.
    
    # Let's refine the reduce function.
    def run():
        input_data = sys.stdin.read().split()
        if not input_data: return
        H, W, Q = map(int, input_data[:3])
        qs = [ (int(input_data[i]), int(input_data[i+1])) for i in range(3, len(input_data), 2) ]
        
        # Initialize: r_walls[r] = [1, 2, ..., W], c_walls[c] = [1, 2, ..., H]
        # Use list comprehensions.
        r_walls = [list(range(1, W + 1)) for _ in range(H)]
        c_walls = [list(range(1, H + 1)) for _ in range(W)]
        
        def destroy(state, q):
            destroyed_set, rw, cw = state
            r, c = q
            
            # Check if wall exists at (r, c)
            # We need to know if (r, c) is still in rw[r-1]
            # Since we can't use a loop, we use bisect to check existence.
            idx_in_row = bisect_left(rw[r-1], c)
            exists = idx_in_row < W and rw[r-1][idx_in_row] == c
            
            if exists:
                # Destroy wall at (r, c)
                # Update rw and cw using slicing
                # Note: rw and cw are lists of lists. We must create a new list for the row.
                # But we can't mutate. We must return a new state.
                # To avoid O(H) copy, we can mutate the list in place and return it.
                # The constraint says "no loops", it doesn't say "no mutation".
                
                # Mutation is allowed.
                row_list = rw[r-1]
                col_list = cw[c-1]
                row_list.pop(idx_in_row)
                
                # For the column list, we need the index of r.
                idx_in_col = bisect_left(col_list, r)
                col_list.pop(idx_in_col)
                
                return (destroyed_set | {(r, c)}, rw, cw)
            else:
                # Destroy 4 neighbors
                # Up
                col_list = cw[c-1]
                idx_c = bisect_left(col_list, r)
                # Wall above is at idx_c - 1
                
                # Down
                # Wall below is at idx_c
                
                # Left
                row_list = rw[r-1]
                idx_r = bisect_left(row_list, c)
                # Wall left is at idx_r - 1
                
                # Right
                # Wall right is at idx_r
                
                # We need to collect all walls to destroy first, then destroy them
                # to avoid index shifts during the process.
                
                targets = []
                if idx_c > 0: targets.append((col_list[idx_c-1], c))
                if idx_c < len(col_list): targets.append((col_list[idx_c], c))
                if idx_r > 0: targets.append((r, row_list[idx_r-1]))
                if idx_r < len(row_list): targets.append((r, row_list[idx_r]))
                
                # Now destroy all targets. Since we can't use a loop, 
                # we use reduce again to destroy the targets.
                def destroy_target(s, t):
                    tr, tc = t
                    # Only destroy if it's actually a wall (though by definition it is)
                    #