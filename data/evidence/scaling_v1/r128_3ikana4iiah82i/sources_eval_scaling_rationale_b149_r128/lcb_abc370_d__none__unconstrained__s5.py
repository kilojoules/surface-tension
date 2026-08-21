import sys
from bisect import bisect_left, bisect_right
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries as pairs of (R, C)
    queries = zip(
        map(int, input_data[3::2]), 
        map(int, input_data[4::2])
    )

    # State: (rows_walls, cols_walls, total_walls)
    # rows_walls[i] is a sorted list of column indices that have walls in row i
    # cols_walls[j] is a sorted list of row indices that have walls in column j
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    
    # We use a dictionary or list to track if a wall at (r, c) is destroyed
    # Since H*W is up to 4e5, we can use a set of destroyed walls
    # However, the problem asks for remaining walls.
    # To avoid loops, we use reduce to process queries.
    
    # To handle the "destroy" operation efficiently without loops, 
    # we need a way to remove elements from sorted lists.
    # Since Python's list.pop(i) is O(N), and we have many removals,
    # we must be careful. But given the constraints and the "no loop" rule,
    # we will use a set to track destroyed walls and filter the sorted lists
    # only when necessary, or use a different approach.
    
    # Actually, the most efficient way to find the "nearest" wall is using 
    # sorted lists and binary search. To "destroy" a wall, we remove it from 
    # the row-list and the col-list.
    
    # Because we cannot use loops, we use a recursive-like structure via reduce.
    # We store the walls in a way that we can remove them.
    # Using a set for 'destroyed' and updating the sorted lists.
    
    # Note: Removing from a list is O(N). With 4e5 elements, this might TLE.
    # But the "no loop" constraint is very strict. 
    # Let's use a SortedList-like approach if possible, but we can't import external libs.
    # We will use the fact that we only need the nearest neighbors.
    
    # Revised State: (row_sets, col_sets, current_wall_count)
    # Using sets for O(1) removal and checking.
    # But sets aren't sorted. We need sorted structures for nearest neighbor.
    # Since we can't use loops, we'll use a dictionary of sets and 
    # accept that we must find the nearest wall by checking 
    # outward from the point if the wall is destroyed.
    # Wait, the "no loop" rule prevents 'while' for searching.
    # Let's use the sorted list approach and accept the O(N) removal, 
    # as it's the only way to implement this without loops or recursion.
    
    # Actually, we can use a Fenwick tree or Segment tree to find the nearest wall,
    # but those require loops to initialize and query.
    # The only way to truly avoid loops is to use map/reduce/comprehensions.
    
    # Let's use the sorted list approach with bisect.
    # To handle removals efficiently, we can't. But we can use a set to 
    # track destroyed walls and just skip them.
    # But "skip them" requires a loop.
    
    # Wait, the problem says "destroy the first walls that appear".
    # This means we need the closest existing wall.
    # If we use a set for destroyed walls, we can't find the "next" wall 
    # without a loop.
    
    # Therefore, we MUST remove the walls from the sorted lists.
    # list.pop(index) is the only way.
    
    def process_query(state, query):
        r, c = query
        row_walls, col_walls, count = state
        
        # Check if wall exists at (r, c)
        # r-1 and c-1 for 0-indexing
        # We use a helper to check existence
        # Since we can't use 'if' in a way that looks like a loop, 
        # we use a conditional expression.
        
        # Check if wall at (r, c) exists
        # We can check if c is in row_walls[r-1]
        # But row_walls[r-1] is a list. Checking 'in' is O(W).
        # Let's use a set for each row and column for O(1) checks.
        
        # To avoid loops, we use a tuple of (row_sets, col_sets, count)
        # and sorted lists for the nearest neighbor search.
        pass

    # Given the constraints and the "no loop" rule, the most viable 
    # implementation is using a combination of sets for existence 
    # and sorted lists for proximity, using reduce to iterate.
    
    # However, removing from sorted lists is O(N). 
    # To optimize, we can use a technique where we don't remove, 
    # but that requires loops to find the next valid wall.
    # The only way to find the nearest wall in O(log N) is a 
    # Segment Tree or Balanced BST, both of which are hard to 
    # implement without loops.
    
    # Let's implement the sorted list + pop approach. 
    # It's the most direct translation of the logic.
    
    def run():
        # Using a dictionary of sets for O(1) lookup
        # and lists for sorted access.
        # State: (row_lists, col_lists, wall_count)
        
        def remove_wall(r, c, r_lists, c_lists, current_count):
            # Remove c from r_lists[r-1] and r from c_lists[c-1]
            # Use bisect to find index for O(log N) then pop for O(N)
            idx_c = bisect_left(r_lists[r-1], c)
            if idx_c < len(r_lists[r-1]) and r_lists[r-1][idx_c] == c:
                r_lists[r-1].pop(idx_c)
                idx_r = bisect_left(c_lists[c-1], r)
                c_lists[c-1].pop(idx_r)
                return current_count - 1
            return current_count

        def handle_query(state, query):
            r, c = query
            r_lists, c_lists, count = state
            
            # If wall exists at (r, c)
            # We check if c is in r_lists[r-1] using bisect
            idx = bisect_left(r_lists[r-1], c)
            if idx < len(r_lists[r-1]) and r_lists[r-1][idx] == c:
                return (r_lists, c_lists, remove_wall(r, c, r_lists, c_lists, count))
            
            # If no wall, destroy 4 neighbors
            # Up
            idx_u = bisect_left(c_lists[c-1], r) - 1
            u_wall = (c_lists[c-1][idx_u], c) if idx_u >= 0 else None
            # Down
            idx_d = bisect_right(c_lists[c-1], r)
            d_wall = (c_lists[c-1][idx_d], c) if idx_d < len(c_lists[c-1]) else None
            # Left
            idx_l = bisect_left(r_lists[r-1], c) - 1
            l_wall = (r, r_lists[r-1][idx_l]) if idx_l >= 0 else None
            # Right
            idx_r = bisect_right(r_lists[r-1], c)
            r_wall = (r, r_lists[r-1][idx_r]) if idx_r < len(r_lists[r-1]) else None
            
            # Destroy all found neighbors
            # We must update the state sequentially
            s1 = (r_lists, c_lists, count)
            s2 = (s1[0], s1[1], remove_wall(*u_wall, *s1) if u_wall else s1[2]) if u_wall else s1
            # This is getting messy. Let's use a helper.
            
            # To avoid loops and recursion, we can use a list of targets 
            # and reduce them.
            targets = [u_wall, d_wall, l_wall, r_wall]
            
            def apply_destructions(st, target):
                if target:
                    return (st[0], st[1], remove_wall(target[0], target[1], st[0], st[1], st[2]))
                return st
            
            return reduce(apply_destructions, targets, (r_lists, c_lists, count))

        # Initial state
        row_lists = [list(range(1, W + 1)) for _ in range(H)]
        col_lists = [list(range(1, H + 1)) for _ in range(W)]
        
        final_state = reduce(handle_query, queries, (row_lists, col_lists, H * W))
        print(final_state[2])

    run()

if __name__ == "__main__":
    solve()