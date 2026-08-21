```python
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

    # We need to track walls in each row and each column.
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is large.
    # However, we only care about rows and columns that are actually referenced in queries.
    # But the problem says we start with a wall in EVERY cell.
    # To avoid O(H*W) initialization, we use a dictionary of sorted lists.
    # We only initialize a row/column list when it's first accessed.
    
    # Actually, the constraint H*W <= 4e5 allows us to use lists if we are careful.
    # But since we can't use loops, we use a dictionary and a helper to get the range.
    
    # Let's redefine: we track "destroyed" walls.
    # But the logic requires finding the "first existing wall".
    # This is easier if we track existing walls.
    # Since we can't loop to initialize, we can't use lists of all walls.
    # Wait, we can use a dictionary where keys are row/col indices and values are 
    # SortedLists of existing walls. But we can't initialize them with range().
    
    # Alternative: Track destroyed walls in a set.
    # To find the first existing wall, we need to check neighbors.
    # But the "first" could be far away.
    # The only way to find the "first" wall without a loop is to maintain 
    # a data structure of existing walls and use binary search.
    
    # Since we can't use loops, we can't use `for i in range(H): rows[i] = ...`
    # But we can use a dictionary and only populate it when a query hits that row/col.
    # However, the "first wall" might be in a row/col that hasn't been queried yet.
    # That wall would still exist.
    
    # Let's use a different approach: 
    # A wall at (r, c) is destroyed if:
    # 1. It was the target of a query (R_q, C_q).
    # 2. It was the first wall in one of the 4 directions from a query (R_q, C_q) 
    #    where (R_q, C_q) was already destroyed.
    
    # This looks like we can simulate it using a functional approach (reduce).
    # We need to maintain the state of the grid.
    # Since we can't use loops, we use a dictionary of sets for destroyed walls.
    # To find the first existing wall, we can't loop. 
    # But we can use the fact that we only need to check the "boundaries" of 
    # destroyed segments.
    
    # Actually, the simplest way to implement this without loops is to use 
    # a dictionary of sorted lists for each row and column, and 
    # initialize them lazily. But we can't use loops to fill them.
    # Wait! We can use a dictionary and for any row/col, if it's not in the dict,
    # we treat it as having all walls.
    
    # Let's use a dictionary `rows` and `cols` where we store the indices of 
    # DESTROYED walls.
    # To find the first existing wall to the right of (r, c):
    # We look at the destroyed segments in row r. 
    # If (r, c) is destroyed, the first existing wall is the first index 
    # after the contiguous block of destroyed walls containing c.
    
    # This is getting complex. Let's reconsider: H*W <= 4e5.
    # We can use a dictionary to store the state of each cell: 1 for wall, 0 for empty.
    # But we can't loop to initialize. 
    # We can use a default value of 1 for the dictionary.
    
    # To find the first wall in a direction without a loop:
    # We can maintain a Disjoint Set Union (DSU) for each row and each column.
    # DSU can tell us the next/previous available wall.
    # Since we can't use loops, we can't initialize DSU arrays.
    # But we can use a dictionary for DSU parents.
    
    # Let's use the "destroyed" set and a helper to find the first wall.
    # Since we can't loop, we can use a recursive-like structure or 
    # a mathematical way to find the gap in the destroyed set.
    
    # Actually, the most straightforward way is to use a dictionary of 
    # sorted lists of destroyed cells for each row and column.
    # To find the first existing wall to the right of c in row r:
    # 1. Find the range of destroyed cells containing c.
    # 2. The first existing wall is the one immediately after that range.
    
    # We can use `bisect_left` to find the position of c in the sorted list of 
    # destroyed cells for row r. Then we check if the neighbors are contiguous.
    # But we still need to find the end of the contiguous block.
    # We can do this by storing destroyed segments in a sorted list of tuples.
    
    # Let's use a simpler approach: 
    # Since we can't use loops, we use `functools.reduce` to process queries.
    # The state is (destroyed_set, row_segments, col_segments).
    # row_segments is a dict: row -> sorted list of (start, end) of destroyed blocks.
    
    from functools import reduce
    from bisect import bisect_right

    def get_first_existing(segments, pos, limit, direction):
        # segments: sorted list of (start, end)
        # direction: 1 for right/down, -1 for left/up
        idx = bisect_right(segments, (pos, float('inf'))) - 1
        if idx >= 0 and segments[idx][0] <= pos <= segments[idx][1]:
            # pos is inside a destroyed block
            block_start, block_end = segments[idx]
            if direction == 1:
                return block_end + 1 if block_end < limit else None
            else:
                return block_start - 1 if block_start > 1 else None
        else:
            # pos is an existing wall
            return pos

    def process_query(state, q):
        destroyed, row_segs, col_segs = state
        r, c = q
        
        if (r, c) not in destroyed:
            # Destroy wall at (r, c)
            # Update segments
            def update_segs(segs, pos, limit):
                # Find if pos is adjacent to any existing blocks
                # This is tricky without loops. 
                # Let's just use a set for destroyed and a helper for the "first"
                # But the helper needs to find the block boundary.
                pass
            
            # Instead of complex segment logic, let's use the fact that 
            # we can afford a bit of overhead.
            # We'll store destroyed cells in a set and use a dictionary of 
            # sorted lists to find the first existing wall.
            # To find the first existing wall to the right of c:
            # We find the contiguous block of destroyed cells starting at c.
            # Since we can't loop, we can use a DSU-like structure in a dictionary.
            pass

    # Given the constraints and the "no loop" rule, the only way to 
    # find the "first" wall is to maintain the boundaries of destroyed 
    # intervals. We can use a dictionary of sorted lists of intervals.
    # To merge intervals without a loop, we can use the fact that 
    # we only merge the interval containing the new cell with its 
    # immediate left and right neighbors.
    
    # Let's refine the state: (destroyed_set, row_intervals, col_intervals)
    # row_intervals: dict of {row: sorted_list_of_intervals}
    
    def merge_and_add(intervals, pos, limit):
        # Find intervals that overlap with [pos, pos]
        # Since they are sorted and non-overlapping, only one can overlap.
        # Then check if the one to the left or right is adjacent.
        # We can use bisect to find the position.
        
        # This is still complex to do without any loops or recursion.
        # Wait, the problem says "no for or while loops". 
        # It doesn't forbid list comprehensions or map/filter/reduce.
        pass

    # Let's use a different strategy. 
    # We can use a dictionary to store the "next" and "prev" wall for each cell.
    # But that's H*W. 
    # However, we only need to track the "destroyed" regions.
    # For each row, we maintain a sorted list of destroyed intervals.
    # When cell (r, c) is destroyed:
    # 1. Find the interval containing c (if any).
    # 2. Merge it with the interval ending at c-1 and the one starting at c+1.
    # 3. Update the sorted list.
    
    # Since we can't use loops, we can use a helper function to handle the 
    # interval merging logic and call it via reduce.
    
    def handle_query(state, q):
        destroyed, row_ints, col_ints = state
        r, c = q
        
        if (r, c) not in destroyed:
            # Case 1: Wall at (r, c) exists, destroy it.
            new_destroyed = destroyed | {(r, c)}
            
            # Update row intervals
            r_list = row_ints.get(r, [])
            # Find if c is adjacent to any existing intervals
            # We can use a list comprehension to find intervals to merge
            to_merge = [intv for intv in r_list if intv[1] == c-1 or intv[0] == c+1 or (intv[0] <= c <= intv[1])]
            # The new interval is (min(starts), max(ends))
            new_int = (min([intv[0] for intv in to_merge] + [c]), 
                       max([intv[1] for intv in to_merge] + [c]))
            # Remove merged intervals and add the new one
            new_r_list = sorted([intv for intv in r_list if intv not in to_merge] + [new_int])
            
            # Update col intervals
            c_list = col_ints.get(c, [])
            to_merge_c = [intv for intv in c_list if intv[1] == r-1 or intv[0] == r+1 or (intv[0] <= r <= intv[1])]
            new_int_c = (min([intv[0] for intv in to_merge_c] + [r]), 
                         max([intv[1] for intv in to_merge_c] + [r]))
            new_c_list = sorted([intv for intv in c_list if intv not in to_merge_c] + [new_int_c])
            
            return (new_destroyed, {**row_ints, r: new_r_list}, {**col_ints, c: new_c_list})
        
        else:
            # Case 2: No wall at (r, c), destroy first walls in 4 directions.
            # We need to find the first existing wall.
            # The first existing wall is the boundary of the destroyed interval containing (r, c).
            
            # Row boundaries
            r_list = row_ints.get(r, [])
            # Find the interval containing c
            curr_r_int = next( (intv for intv in r_list if intv[0] <= c <= intv[1]), (c, c) )
            # The first existing walls are at curr_r_int[0] - 1 and curr_r_int[1] + 1
            # But only if they are within [1, W]
            targets_r = [ (r, curr_r_int[0] - 1), (r, curr_r_int[1] + 1) ]
            
            # Col boundaries
            c_list = col_ints.get(c, [])
            curr_c_int =