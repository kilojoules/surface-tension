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
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # We maintain sets of existing wall indices for each row and each column.
    # rows[i] contains all j such that (i, j) has a wall.
    # cols[j] contains all i such that (i, j) has a wall.
    # Using lists and bisect for efficient searching.
    
    # Initial state: all cells have walls.
    # To avoid explicit loops, we use list comprehensions.
    rows = [list(range(1, W + 1)) for _ in range(H)]
    cols = [list(range(1, H + 1)) for _ in range(W)]

    def destroy(r, c, state):
        r_idx, c_idx = r - 1, c - 1
        row_list = state[0][r_idx]
        col_list = state[1][c_idx]
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in O(log N)
        pos = bisect_left(row_list, c)
        if pos < len(row_list) and row_list[pos] == c:
            # Wall exists, destroy it
            row_list.pop(pos)
            # Also remove from the corresponding column list
            col_pos = bisect_left(col_list, r)
            col_list.pop(col_pos)
            return state
        
        # No wall at (r, c), destroy 4 neighbors
        # 1. Up and Down (search in col_list)
        c_pos = bisect_left(col_list, r)
        
        # Up: the element just before c_pos
        # Down: the element at c_pos
        
        # 2. Left and Right (search in row_list)
        r_pos = bisect_left(row_list, c)
        
        # Left: the element just before r_pos
        # Right: the element at r_pos

        # To handle simultaneous destruction, we identify all targets first
        targets = []
        # Up
        if c_pos > 0:
            targets.append((col_list[c_pos-1], c))
        # Down
        if c_pos < len(col_list):
            targets.append((col_list[c_pos], c))
        # Left
        if r_pos > 0:
            targets.append((r, row_list[r_pos-1]))
        # Right
        if r_pos < len(row_list):
            targets.append((r, row_list[r_pos]))

        # Destroy all identified targets
        def remove_wall(s, target):
            tr, tc = target
            # Remove from row list
            r_l = s[0][tr-1]
            p_r = bisect_left(r_l, tc)
            if p_r < len(r_l) and r_l[p_r] == tc:
                r_l.pop(p_r)
            # Remove from col list
            c_l = s[1][tc-1]
            p_c = bisect_left(c_l, tr)
            if p_c < len(c_l) and c_l[p_c] == tr:
                c_l.pop(p_c)
            return s

        return reduce(remove_wall, targets, state)

    # Process all queries using reduce
    final_state = reduce(lambda s, q: destroy(q[0], q[1], s), queries, (rows, cols))
    
    # Calculate remaining walls
    # The number of remaining walls is the sum of lengths of all row lists
    total_remaining = sum(map(len, final_state[0]))
    print(total_remaining)

if __name__ == "__main__":
    solve()