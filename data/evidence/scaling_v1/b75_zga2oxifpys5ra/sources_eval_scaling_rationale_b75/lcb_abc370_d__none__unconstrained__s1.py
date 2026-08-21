import sys
from bisect import bisect_left
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

    # State: (row_walls, col_walls, total_walls)
    # row_walls[i] is a sorted list of columns that have a wall in row i
    # col_walls[j] is a sorted list of rows that have a wall in column j
    initial_state = (
        [list(range(1, W + 1)) for _ in range(H)],
        [list(range(1, H + 1)) for _ in range(W)],
        H * W
    )

    def remove_wall(state, r, c):
        row_walls, col_walls, total = state
        # Use bisect to find indices for removal
        # Note: r and c are 1-indexed
        r_idx, c_idx = r - 1, c - 1
        
        # Find position of c in row_walls[r_idx]
        # Since we need to remove it, we find the index and pop
        # However, we cannot use loops/mutating methods in a way that 
        # violates the spirit, but we must mutate the lists for performance.
        # The prompt forbids 'for' and 'while' loops, not mutation of objects.
        
        # Check if wall exists
        # We use a helper to check existence and remove
        def try_remove(sorted_list, val):
            idx = bisect_left(sorted_list, val)
            if idx < len(sorted_list) and sorted_list[idx] == val:
                sorted_list.pop(idx)
                return True
            return False

        # This is a helper to handle the logic of one wall destruction
        # We wrap it in a function to avoid loops
        def destroy(s, row, col):
            rw, cw, t = s
            if try_remove(rw[row-1], col) and try_remove(cw[col-1], row):
                return (rw, cw, t - 1)
            return (rw, cw, t)

        return destroy(state, r, c)

    def process_query(state, query):
        r, c = query
        row_walls, col_walls, total = state
        
        # Check if wall exists at (r, c)
        # We check if c is in the sorted list of row_walls[r-1]
        idx = bisect_left(row_walls[r-1], c)
        exists = (idx < len(row_walls[r-1]) and row_walls[r-1][idx] == c)
        
        if exists:
            # Destroy wall at (r, c)
            # We use a helper to remove and return new state
            def remove_single(s, row, col):
                rw, cw, t = s
                # Use bisect to find and pop
                i_r = bisect_left(rw[row-1], col)
                rw[row-1].pop(i_r)
                i_c = bisect_left(cw[col-1], row)
                cw[col-1].pop(i_c)
                return (rw, cw, t - 1)
            return remove_single(state, r, c)
        else:
            # Destroy 4 nearest walls
            rw, cw, t = state
            
            # Up
            idx_u = bisect_left(cw[c-1], r) - 1
            # Down
            idx_d = bisect_left(cw[c-1], r)
            # Left
            idx_l = bisect_left(rw[r-1], c) - 1
            # Right
            idx_r = bisect_left(rw[r-1], c)
            
            # Collect targets
            targets = []
            if idx_u >= 0: targets.append((cw[c-1][idx_u], c))
            if idx_d < len(cw[c-1]): targets.append((r_val := cw[c-1][idx_d], c))
            if idx_l >= 0: targets.append((r, c_val := rw[r-1][idx_l]))
            if idx_r < len(rw[r-1]): targets.append((r, c_val2 := rw[r-1][idx_r]))
            
            # To avoid loops, we use map/reduce to remove targets
            def remove_target(s, target):
                tr, tc = target
                # Check if wall still exists (since targets can overlap)
                # We must check existence because one query can destroy the same wall twice
                # (e.g., the wall above is the same as the wall to the left)
                # However, the problem says "simultaneously", so we identify targets first.
                # But we must ensure we don't double-count the reduction of 'total'.
                
                # Because we need to remove them, and the prompt forbids loops,
                # we use a helper function and reduce.
                # But wait, the targets are identified SIMULTANEOUSLY.
                # We should use a set of coordinates to avoid double-removing.
                return s

            # Correct logic for simultaneous destruction:
            # 1. Identify all 4 potential coordinates.
            # 2. Remove duplicates.
            # 3. Remove each from the grid.
            
            # Since we can't use loops, we use reduce to apply the removal
            # We redefine the removal logic inside process_query
            def apply_removals(s, target_list):
                curr_s = s
                # Use a helper to remove a specific wall if it exists
                def rem(st, target):
                    tr, tc = target
                    rw_s, cw_s, t_s = st
                    # Check if wall exists
                    i_r = bisect_left(rw_s[tr-1], tc)
                    if i_r < len(rw_s[tr-1]) and rw_s[tr-1][i_r] == tc:
                        rw_s[tr-1].pop(i_r)
                        i_c = bisect_left(cw_s[tc-1], tr)
                        cw_s[tc-1].pop(i_c)
                        return (rw_s, cw_s, t_s - 1)
                    return st
                
                return reduce(rem, target_list, curr_s)

            # Use a set to handle simultaneous destruction of the same wall
            unique_targets = list(set(targets))
            return apply_removals(state, unique_targets)

    # Process all queries using reduce
    final_state = reduce(process_query, queries, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()