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
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # We maintain a list of sorted indices of existing walls for every row and column.
    # Initially, every cell has a wall.
    # rows_walls[i] contains sorted indices of columns that have a wall in row i.
    # cols_walls[j] contains sorted indices of rows that have a wall in column j.
    # Since H*W is up to 4e5, we can't use a simple list of sets if we want to avoid loops.
    # However, we can use a dictionary of sorted lists (or similar) and update them.
    # But wait, updating a sorted list (removing an element) is O(N). 
    # Given the constraints and the "no loop" rule, we must use a strategy that 
    # fits into reduce(). 
    # Actually, the most efficient way to find the nearest wall is using a SortedList,
    # but that's not built-in. With built-ins and no loops, we can use sets and 
    # bisect on sorted lists, but updating the sorted lists requires rebuilding 
    # or using a data structure that supports O(log N) deletion.
    # Since we cannot use loops, we will use a dictionary of sets to track 
    # destroyed walls and use a custom function within reduce to handle the logic.
    
    # To avoid explicit loops and recursion, we use functools.reduce to process queries.
    from functools import reduce

    # State: (destroyed_set, row_walls, col_walls)
    # row_walls: list of sets, col_walls: list of sets
    # But sets aren't sorted. We need sorted structures.
    # Given the "no loop" constraint, we can't easily maintain SortedLists.
    # Let's use a different approach: track destroyed cells in a set.
    # To find the nearest wall, we can't iterate. But we can use a 
    # mathematical approach or a very clever use of filter/map.
    # Actually, the only way to find the "nearest" wall without a loop 
    # is to maintain a sorted list of existing walls and use bisect.
    # To "remove" an element from a sorted list without a loop, 
    # we can't use .pop(i) in a loop, but we can use slice assignment 
    # or create a new list. However, creating a new list is O(N).
    # With H*W = 4e5, O(N) per query is too slow.
    
    # Wait, the constraint says "no for/while loops". 
    # We can use list comprehensions, map, filter, and reduce.
    # We can use a Fenwick tree or Segment tree implemented via 
    # list comprehensions? No, that's impossible for updates.
    # But we can use a dictionary to simulate a Balanced BST or 
    # just use the fact that we can use 'set' and 'bisect' if we 
    # maintain sorted lists and update them using slice notation.
    # slice notation `l[i:i+1] = []` is a way to delete.
    
    # Let's refine: 
    # We need to find the nearest wall. If we maintain sorted lists of 
    # existing walls for each row and column, we can use bisect_left.
    # To delete, we use `l.pop(index)`. 
    # Since we can't use for-loops, we use reduce to iterate over queries.
    
    # To handle the "no loop" constraint strictly, we use a helper 
    # function and reduce.
    
    def process_query(state, query):
        r, c = query
        destroyed, row_walls, col_walls = state
        
        if (r, c) in destroyed:
            # Destroy 4 nearest walls
            # Up
            col_list = col_walls[c]
            idx = bisect_left(col_list, r)
            # Wall above is at idx-1
            up_wall = (col_list[idx-1], c) if idx > 0 else None
            # Down
            down_wall = (col_list[idx], c) if idx < len(col_list) else None
            
            # Left
            row_list = row_walls[r]
            idx_c = bisect_left(row_list, c)
            # Wall left is at idx_c-1
            left_wall = (r, row_list[idx_c-1]) if idx_c > 0 else None
            # Right
            right_wall = (r, row_list[idx_c],) if idx_c < len(row_list) else None
            
            targets = [t for t in [up_wall, down_wall, left_wall, right_wall] if t]
            
            # Update state: remove targets from row_walls and col_walls
            # Since we can't loop, we use a nested reduce or a map to remove targets
            def remove_wall(s, wall):
                if wall is None: return s
                wr, wc = wall
                s[0].add(wall)
                # Remove from sorted lists
                # We use the fact that we can call .pop() inside a function
                # and use map/reduce to apply it to all targets.
                r_list = s[1][wr]
                r_idx = bisect_left(r_list, wc)
                if r_idx < len(r_list) and r_list[r_idx] == wc:
                    r_list.pop(r_idx)
                c_list = s[2][wc]
                c_idx = bisect_left(c_list, wr)
                if c_idx < len(c_list) and c_list[c_idx] == wr:
                    c_list.pop(c_idx)
                return s

            return reduce(remove_wall, targets, (destroyed, row_walls, col_walls))
        else:
            # Destroy wall at (r, c)
            destroyed.add((r, c))
            r_list = row_walls[r]
            r_idx = bisect_left(r_list, c)
            r_list.pop(r_idx)
            c_list = col_walls[c]
            c_idx = bisect_left(c_list, r)
            c_list.pop(c_idx)
            return (destroyed, row_walls, col_walls)

    # Initialize state
    # Using 1-based indexing for convenience
    initial_row_walls = {i: list(range(1, W + 1)) for i in range(1, H + 1)}
    initial_col_walls = {j: list(range(1, H + 1)) for j in range(1, W + 1)}
    initial_state = (set(), initial_row_walls, initial_col_walls)
    
    final_state = reduce(process_query, queries, initial_state)
    
    # Result is total walls - destroyed walls
    print(H * W - len(final_state[0]))

solve()