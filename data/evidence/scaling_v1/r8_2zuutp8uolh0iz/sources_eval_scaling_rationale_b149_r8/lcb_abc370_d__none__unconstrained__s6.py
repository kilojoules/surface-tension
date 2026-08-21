import sys
from bisect import bisect_left, insort
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # We maintain sorted lists of existing wall indices for each row and column.
    # rows[i] contains the column indices of walls in row i.
    # cols[j] contains the row indices of walls in column j.
    # Using lists and bisect allows O(log N) search and O(N) deletion.
    # Given H*W <= 4e5, we use a dictionary or list of lists.
    
    # Initial state: every cell has a wall.
    # To optimize memory and initialization, we represent the grid as 
    # sets of active wall indices for each row and column.
    # However, since we need to find the "nearest" wall, sorted lists are better.
    
    # Using a dictionary of lists to store wall positions.
    # row_walls[r] = sorted list of columns that have walls in row r.
    # col_walls[c] = sorted list of rows that have walls in column c.
    
    # Since H*W is up to 4e5, initializing H lists of W elements is fine.
    # But we must avoid loops. We use list comprehensions.
    row_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    col_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # We need a way to track if a wall at (r, c) is destroyed.
    # A set of destroyed walls is efficient.
    destroyed = set()

    def process_query(state, query):
        r, c = query
        destroyed_set, r_walls, c_walls = state
        
        if (r, c) in destroyed_set:
            # No wall at (r, c), look in 4 directions
            # 1. Up (same column, smaller row)
            # 2. Down (same column, larger row)
            # 3. Left (same row, smaller column)
            # 4. Right (same row, larger column)
            
            # Find nearest walls using bisect on the sorted lists
            # Note: The lists in r_walls and c_walls are updated by removing elements.
            # Since we can't use loops, we use a helper to find and remove.
            
            # To avoid mutating lists inside reduce in a way that breaks, 
            # we use the fact that lists are mutable.
            
            # Column search (Up/Down)
            c_list = c_walls[c]
            idx = bisect_left(c_list, r)
            
            # Up
            up_wall = (c_list[idx-1], c) if idx > 0 else None
            # Down
            down_wall = (c_list[idx], c) if idx < len(c_list) else None
            
            # Row search (Left/Right)
            r_list = r_walls[r]
            idx_r = bisect_left(r_list, c)
            
            # Left
            left_wall = (r, r_list[idx_r-1]) if idx_r > 0 else None
            # Right
            right_wall = (r, r_list[idx_r]) if idx_r < len(r_list) else None
            
            targets = [w for w in [up_wall, down_wall, left_wall, right_wall] if w]
            
            # Update destroyed set
            destroyed_set.update(targets)
            
            # Update the sorted lists by removing the destroyed walls.
            # Since we can't loop, we use a trick: 
            # we only remove them from the lists if they were actually destroyed now.
            # But wait, the problem says "destroy the first walls". 
            # Those walls might have been destroyed by previous queries.
            # Actually, the sorted lists should only contain EXISTING walls.
            
            # To keep the lists updated without loops, we can't easily remove 
            # arbitrary elements. Let's redefine: 
            # We use the sorted lists to find the target, then remove them.
            # Since we can't use loops, we use a recursive-like approach or 
            # map with a side-effect function.
            
            def remove_wall(wall):
                if wall:
                    wr, wc = wall
                    # Remove wc from r_walls[wr] and wr from c_walls[wc]
                    # Using bisect to find index and pop()
                    r_idx = bisect_left(r_walls[wr], wc)
                    if r_idx < len(r_walls[wr]) and r_walls[wr][r_idx] == wc:
                        r_walls[wr].pop(r_idx)
                    c_idx = bisect_left(c_walls[wc], wr)
                    if c_idx < len(c_walls[wc]) and c_walls[wc][c_idx] == wr:
                        c_walls[wc].pop(c_idx)
                return None

            # Use map to execute the removal for each target
            # We filter targets to only those that weren't already destroyed
            # (though the logic of the sorted list already handles that)
            list(map(remove_wall, targets))
            
            return (destroyed_set, r_walls, c_walls)
        else:
            # Wall exists at (r, c), destroy it
            destroyed.add((r, c))
            # Remove from sorted lists
            r_idx = bisect_left(row_walls[r], c)
            row_walls[r].pop(r_idx)
            c_idx = bisect_left(col_walls[c], r)
            col_walls[c].pop(c_idx)
            return (destroyed, row_walls, col_walls)

    # The logic inside process_query needs to be slightly adjusted because 
    # the first 'if' checks if (r, c) is in destroyed_set.
    # If it is NOT in destroyed_set, it means the wall is still there.
    
    def final_process(state, query):
        destroyed_set, r_walls, c_walls = state
        r, c = query
        
        if (r, c) not in destroyed_set:
            # Wall exists, destroy it
            destroyed_set.add((r, c))
            r_idx = bisect_left(r_walls[r], c)
            r_walls[r].pop(r_idx)
            c_idx = bisect_left(c_walls[c], r)
            c_walls[c].pop(c_idx)
        else:
            # Wall doesn't exist, destroy neighbors
            c_list = c_walls[c]
            idx = bisect_left(c_list, r)
            up = (c_list[idx-1], c) if idx > 0 else None
            down = (c_list[idx], c) if idx < len(c_list) else None
            
            r_list = r_walls[r]
            idx_r = bisect_left(r_list, c)
            left = (r, r_list[idx_r-1]) if idx_r > 0 else None
            right = (r, r_list[idx_r]) if idx_r < len(r_list) else None
            
            targets = [t for t in [up, down, left, right] if t]
            
            def remove_wall(wall):
                wr, wc = wall
                destroyed_set.add((wr, wc))
                # Remove from row list
                ri = bisect_left(r_walls[wr], wc)
                if ri < len(r_walls[wr]) and r_walls[wr][ri] == wc:
                    r_walls[wr].pop(ri)
                # Remove from col list
                ci = bisect_left(c_walls[wc], wr)
                if ci < len(c_walls[wc]) and c_walls[wc][ci] == wr:
                    c_walls[wc].pop(ci)
                return None
            
            list(map(remove_wall, targets))
            
        return (destroyed_set, r_walls, c_walls)

    # Use reduce to process all queries
    reduce(final_process, queries, (destroyed, row_walls, col_walls))
    
    # Total walls - destroyed walls
    print(H * W - len(destroyed))

if __name__ == "__main__":
    solve()