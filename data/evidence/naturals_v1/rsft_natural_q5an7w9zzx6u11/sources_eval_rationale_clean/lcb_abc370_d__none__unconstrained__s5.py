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

    # State consists of:
    # rows: A list of sorted lists containing indices of existing walls for each row
    # cols: A list of sorted lists containing indices of existing walls for each column
    # total_walls: Current count of walls
    
    # Initialize rows and cols
    # Using list comprehensions to avoid explicit for-loops
    initial_rows = [list(range(1, W + 1)) for _ in range(H)]
    initial_cols = [list(range(1, H + 1)) for _ in range(W)]
    initial_state = (initial_rows, initial_cols, H * W)

    def process_query(state, query):
        rows, cols, count = state
        r, c = query
        
        # Check if wall exists at (r, c)
        # We use bisect to check existence in O(log N)
        r_idx = r - 1
        c_idx = c - 1
        
        # Check if c is in rows[r_idx]
        pos = bisect_left(rows[r_idx], c)
        has_wall = pos < len(rows[r_idx]) and rows[r_idx][pos] == c
        
        if has_wall:
            # Destroy wall at (r, c)
            # Update rows and cols
            # Note: list.pop(pos) is used; while it's O(N), 
            # the constraints and Python's list implementation 
            # often allow this for 4e5 total elements.
            # However, to be strictly loop-free and functional, 
            # we update the lists.
            rows[r_idx].pop(pos)
            
            # Find position in column list to pop
            col_pos = bisect_left(cols[c_idx], r)
            cols[c_idx].pop(col_pos)
            
            return (rows, cols, count - 1)
        else:
            # Destroy 4 neighbors
            # We need to find the elements immediately to the left, right, top, bottom
            
            # Row neighbors (Left and Right)
            # Left: element at pos-1, Right: element at pos
            # Column neighbors (Up and Down)
            # Up: element at col_pos-1, Down: element at col_pos
            
            col_pos = bisect_left(cols[c_idx], r)
            
            # Identify targets
            targets = []
            # Left
            if pos > 0: targets.append((r, rows[r_idx][pos-1]))
            # Right
            if pos < len(rows[r_idx]): targets.append((r, rows[r_idx][pos]))
            # Up
            if col_pos > 0: targets.append((cols[c_idx][col_pos-1], c))
            # Down
            if col_pos < len(cols[c_idx]): targets.append((cols[c_idx][col_pos], c))
            
            # To avoid loops, we use reduce to remove all identified targets from the state
            def remove_wall(s, target):
                tr, tc = target
                sr, sc, cnt = s
                
                # Check if wall still exists (targets might overlap)
                t_row_idx = tr - 1
                t_col_idx = tc - 1
                t_pos = bisect_left(sr[t_row_idx], tc)
                
                if t_pos < len(sr[t_row_idx]) and sr[t_row_idx][t_pos] == tc:
                    sr[t_row_idx].pop(t_pos)
                    t_col_pos = bisect_left(sc[t_col_idx], tr)
                    sc[t_col_idx].pop(t_col_pos)
                    return (sr, sc, cnt - 1)
                return s

            return reduce(remove_wall, targets, (rows, cols, count))

    final_state = reduce(process_query, queries, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()