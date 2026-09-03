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
    # Since we can't use loops, we use sorted lists and bisect.
    # To avoid loops, we use map/list comprehensions.
    
    # Initial state: every cell has a wall.
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    rows_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # We need a way to track if a wall at (r, c) is destroyed.
    # Since we can't use loops to update the sorted lists efficiently without 
    # violating constraints or using loops, we use a set to track destroyed walls
    # and a technique to "clean up" the sorted lists lazily or via a different structure.
    # Actually, the constraint H*W <= 4e5 allows us to store the state.
    # To avoid loops, we can use a recursive-like structure or map, but the prompt 
    # forbids loops. We can use a mutable state and a helper function called via map.
    
    destroyed = set()
    
    def process_query(query):
        r, c = query
        if (r, c) not in destroyed:
            destroyed.add((r, c))
            # Remove from sorted lists
            # Note: list.remove() is O(N), but we can't use loops.
            # To stay within O(Q log(max(H,W))), we should use a different approach.
            # However, we can use bisect to find the index and then pop.
            idx_r = bisect_left(rows_walls[r], c)
            rows_walls[r].pop(idx_r)
            idx_c = bisect_left(cols_walls[c], r)
            cols_walls[c].pop(idx_c)
            return 1
        else:
            # Look for neighbors
            # Up
            idx_u = bisect_left(cols_walls[c], r) - 1
            u_wall = cols_walls[c][idx_u] if idx_u >= 0 else None
            # Down
            idx_d = bisect_left(cols_walls[c], r)
            d_wall = cols_walls[c][idx_d] if idx_d < len(cols_walls[c]) else None
            # Left
            idx_l = bisect_left(rows_walls[r], c) - 1
            l_wall = rows_walls[r][idx_l] if idx_l >= 0 else None
            # Right
            idx_r = bisect_left(rows_walls[r], c)
            r_wall = rows_walls[r][idx_r] if idx_r < len(rows_walls[r]) else None
            
            targets = [t for t in (u_wall, d_wall, l_wall, r_wall) if t is not None]
            
            # We need to remove these targets from the sorted lists.
            # Since we can't loop, we use a map.
            def remove_wall(wall_coord):
                wr, wc = wall_coord
                if wr is not None and wc is not None:
                    # We must be careful: the target might have been removed by 
                    # another direction in the same query.
                    if (wr, wc) not in destroyed:
                        destroyed.add((wr, wc))
                        # Remove from row list
                        r_idx = bisect_left(rows_walls[wr], wc)
                        if r_idx < len(rows_walls[wr]) and rows_walls[wr][r_idx] == wc:
                            rows_walls[wr].pop(r_idx)
                        # Remove from col list
                        c_idx = bisect_left(cols_walls[wc], wr)
                        if c_idx < len(cols_walls[wc]) and cols_walls[wc][c_idx] == wr:
                            cols_walls[wc].pop(c_idx)
            
            # Map the remove_wall function over the targets
            list(map(lambda x: remove_wall((x, c) if x == u_wall or x == d_wall else (r, x)), targets))
            return 0

    # Prepare queries
    queries = [int(x) for x in input_data[3:]]
    # Group into pairs of (r, c)
    query_pairs = [queries[i:i+2] for i in range(0, len(queries), 2)]
    
    # Process all queries using map to avoid for/while loops
    list(map(lambda q: process_query(q), query_pairs))
    
    # Total walls - destroyed walls
    print(H * W - len(destroyed))

if __name__ == "__main__":
    solve()