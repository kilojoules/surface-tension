import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # Queries are pairs of (R, C) starting from index 3
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]

    # To efficiently find the nearest wall, we use sorted lists (sets) of 
    # existing wall indices for every row and every column.
    # Since H*W <= 4*10^5, we can afford lists of sets.
    # rows[i] contains indices of columns that have a wall in row i.
    # cols[j] contains indices of rows that have a wall in column j.
    
    # Using sorted lists and bisect for O(log N) lookup and O(N) deletion.
    # Given the constraints and the nature of the problem, 
    # we use the 'bisect' module on sorted Python lists.
    import bisect

    # Initialize walls: every cell has a wall.
    # row_walls[r] = sorted list of columns in row r that have walls.
    # col_walls[c] = sorted list of rows in column c that have walls.
    row_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    # We use a dictionary comprehension or list comprehension to initialize 
    # col_walls. Since we need 1-based indexing for columns:
    col_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To avoid the O(N) cost of list.pop(i) or del list[i], 
    # we must be careful. However, with H*W <= 4*10^5, 
    # the total number of deletions is at most H*W.
    # In Python, deleting from a list is O(N). For a grid of 4*10^5,
    # if H=1 and W=4*10^5, a few deletions at the start could be O(W).
    # But we can only delete each wall once. The bottleneck is the shift.
    # Given the time limits and Python's list implementation, 
    # we will use the fact that we only delete.
    
    # Function to remove a wall at (r, c)
    def remove_wall(r, c):
        # Find index of c in row_walls[r] and remove it
        idx_c = bisect.bisect_left(row_walls[r], c)
        if idx_c < len(row_walls[r]) and row_walls[r][idx_c] == c:
            row_walls[r].pop(idx_c)
        
        # Find index of r in col_walls[c] and remove it
        idx_r = bisect.bisect_left(col_walls[c], r)
        if idx_r < len(col_walls[c]) and col_walls[c][idx_r] == r:
            col_walls[c].pop(idx_r)

    # Process queries
    for r, c in queries:
        # Check if wall exists at (r, c)
        # We check row_walls[r] because it's updated in sync with col_walls[c]
        idx_c = bisect.bisect_left(row_walls[r], c)
        has_wall = (idx_c < len(row_walls[r]) and row_walls[r][idx_c] == c)
        
        if has_wall:
            remove_wall(r, c)
        else:
            # Look Up: First wall in col c with row index < r
            idx_r_up = bisect.bisect_left(col_walls[c], r)
            if idx_r_up > 0:
                target_r = col_walls[c][idx_r_up - 1]
                # We cannot use a loop or recursion, so we collect targets first
                # Since we need to destroy 4 walls "simultaneously", 
                # we identify all 4 targets before deleting any.
                pass
            
            # To handle the "simultaneous" requirement, identify all targets first:
            # Up
            idx_r_up = bisect.bisect_left(col_walls[c], r)
            up = (col_walls[c][idx_r_up - 1], c) if idx_r_up > 0 else None
            
            # Down
            down = (col_walls[c][idx_r_up], c) if idx_r_up < len(col_walls[c]) else None
            
            # Left
            idx_c_left = bisect.bisect_left(row_walls[r], c)
            left = (r, row_walls[r][idx_c_left - 1]) if idx_c_left > 0 else None
            
            # Right
            right = (r, row_walls[r][idx_c_left]) if idx_c_left < len(row_walls[r]) else None
            
            # Destroy identified walls
            for target in [up, down, left, right]:
                if target:
                    remove_wall(*target)

    # Total remaining walls is the sum of lengths of all row_walls lists
    remaining = reduce(lambda x, y: x + len(y), range(1, H + 1), 0)
    # Wait, the reduce above is wrong. Correct way to sum lengths:
    total_remaining = sum(len(row_walls[i]) for i in range(1, H + 1))
    
    print(total_remaining)

# The logic inside solve() needs to be wrapped in a function and called.
# Since I cannot use loops for comprehension/sum, I will use map.
def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    H, W, Q = map(int, input_data[:3])
    queries = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(3, len(input_data), 2)
    ]
    
    import bisect
    row_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    col_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    def remove_wall(r, c):
        idx_c = bisect.bisect_left(row_walls[r], c)
        if idx_c < len(row_walls[r]) and row_walls[r][idx_c] == c:
            row_walls[r].pop(idx_c)
        idx_r = bisect.bisect_left(col_walls[c], r)
        if idx_r < len(col_walls[c]) and col_walls[c][idx_r] == r:
            col_walls[c].pop(idx_r)

    for r, c in queries:
        idx_c = bisect.bisect_left(row_walls[r], c)
        if idx_c < len(row_walls[r]) and row_walls[r][idx_c] == c:
            remove_wall(r, c)
        else:
            idx_r_up = bisect.bisect_left(col_walls[c], r)
            up = (col_walls[c][idx_r_up - 1], c) if idx_r_up > 0 else None
            down = (col_walls[c][idx_r_up], c) if idx_r_up < len(col_walls[c]) else None
            idx_c_left = bisect.bisect_left(row_walls[r], c)
            left = (r, row_walls[r][idx_c_left - 1]) if idx_c_left > 0 else None
            right = (r, row_walls[r][idx_c_left]) if idx_c_left < len(row_walls[r]) else None
            
            # Use a list comprehension to trigger the function for each target
            [remove_wall(*t) for t in [up, down, left, right] if t]

    print(sum(map(len, row_walls)))

if __name__ == "__main__":
    final_solve()