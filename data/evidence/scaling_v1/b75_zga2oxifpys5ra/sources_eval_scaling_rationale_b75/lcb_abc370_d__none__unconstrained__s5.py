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

    # We maintain the state of walls using sorted lists for each row and column.
    # rows[i] contains the column indices of walls in row i.
    # cols[j] contains the row indices of walls in column j.
    # Using lists and bisect allows O(log N) search and O(N) deletion.
    # Given H*W <= 4e5, the total number of deletions is at most H*W.
    
    initial_state = {
        'rows': [list(range(1, W + 1)) for _ in range(H + 1)],
        'cols': [list(range(1, H + 1)) for _ in range(W + 1)],
        'total_walls': H * W
    }

    def process_query(state, query):
        r, c = query
        rows, cols = state['rows'], state['cols']
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in the sorted list
        idx_in_row = bisect_left(rows[r], c)
        wall_exists = idx_in_row < len(rows[r]) and rows[r][idx_in_row] == c
        
        if wall_exists:
            # Destroy wall at (r, c)
            rows[r].pop(idx_in_row)
            # Find and remove r from cols[c]
            idx_in_col = bisect_left(cols[c], r)
            cols[c].pop(idx_in_col)
            return {**state, 'rows': rows, 'cols': cols, 'total_walls': state['total_walls'] - 1}
        
        # No wall at (r, c), destroy 4 nearest walls
        # 1. Up (same column, row < r)
        # 2. Down (same column, row > r)
        # 3. Left (same row, col < c)
        # 4. Right (same row, col > c)
        
        # Find indices for column search
        col_list = cols[c]
        idx_c = bisect_left(col_list, r)
        
        # Targets to destroy: (row, col)
        targets = []
        # Up
        if idx_c > 0: targets.append((col_list[idx_c - 1], c))
        # Down
        if idx_c < len(col_list): targets.append((col_list[idx_c], c))
        
        # Find indices for row search
        row_list = rows[r]
        idx_r = bisect_left(row_list, c)
        
        # Left
        if idx_r > 0: targets.append((r, row_list[idx_r - 1]))
        # Right
        if idx_r < len(row_list): targets.append((r, row_list[idx_r]))
        
        # To avoid mutating state inside a comprehension/map, we use a helper 
        # to remove the walls. Since we can't use loops, we use reduce again.
        def remove_wall(s, target):
            tr, tc = target
            # Verify wall still exists (simultaneous destruction means one query 
            # could target the same wall twice, e.g., the cell above and to the left)
            # However, the problem says "simultaneously", meaning we identify 
            # the 4 targets first, then remove them.
            # We must check if the wall exists because the 4 targets might overlap.
            
            # We use a set for targets to handle overlaps, but the logic requires 
            # checking if the wall is still there.
            # Actually, the simplest way is to use a set of targets and 
            # then filter them.
            return s

        # Because we cannot use loops, we handle the target removal by 
        # mapping a removal function over the unique targets.
        unique_targets = set(targets)
        
        def perform_removal(s, target):
            tr, tc = target
            # Check if wall exists
            r_list = s['rows'][tr]
            c_list = s['cols'][tc]
            i_r = bisect_left(r_list, tc)
            if i_r < len(r_list) and r_list[i_r] == tc:
                r_list.pop(i_r)
                i_c = bisect_left(c_list, tr)
                c_list.pop(i_c)
                return {**s, 'total_walls': s['total_walls'] - 1}
            return s

        final_state = reduce(perform_removal, unique_targets, state)
        return final_state

    result = reduce(process_query, queries, initial_state)
    print(result['total_walls'])

if __name__ == "__main__":
    solve()