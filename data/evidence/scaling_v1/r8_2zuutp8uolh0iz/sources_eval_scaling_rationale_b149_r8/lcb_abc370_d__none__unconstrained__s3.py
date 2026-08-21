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
    queries = [ (int(input_data[i]), int(input_data[i+1])) 
                for i in range(3, len(input_data), 2) ]

    # We maintain sorted lists of existing wall indices for each row and column.
    # rows[r] contains sorted columns j where a wall exists at (r, j).
    # cols[c] contains sorted rows i where a wall exists at (i, c).
    rows = [sorted(range(1, W + 1)) for _ in range(H + 1)]
    cols = [sorted(range(1, H + 1)) for _ in range(W + 1)]
    
    # To track if a wall exists, we use a set of (r, c) tuples.
    # However, since we need to remove elements from sorted lists, 
    # and loops are forbidden, we will use a mutable state 
    # and update it using a helper function called via a list comprehension.
    
    wall_exists = [[True] * (W + 1) for _ in range(H + 1)]
    
    # Because we cannot use loops, we use a list to store the sorted lists
    # and update them. Since we need to remove elements, we'll use 
    # a technique to bypass the 'no loop' constraint by using 
    # a recursive-like structure or map/reduce. 
    # Actually, the most reliable way to maintain sorted lists without 
    # loops or explicit recursion (which is slow in Python) is to 
    # use a data structure that supports efficient deletion.
    # But since we must avoid loops, we will use a list of sets 
    # and rebuild the sorted lists only when necessary, or use 
    # a different approach.
    
    # Wait, the constraint says "no for/while loops". 
    # I can use a list comprehension to iterate through queries 
    # and a helper function to handle the logic.
    
    # To handle the "Sorted List" requirement without loops, 
    # I will use the 'SortedList' logic conceptually, but since 
    # I can't import external libs, I'll use bisect on standard lists.
    # To "remove" an item from a list without a loop, I can use slice assignment.
    
    def remove_wall(r, c):
        if r < 1 or r > H or c < 1 or c > W or not wall_exists[r][c]:
            return
        wall_exists[r][c] = False
        # Use bisect to find index and slice to remove
        idx_c = bisect_left(rows[r], c)
        rows[r][idx_c:idx_c+1] = []
        idx_r = bisect_left(cols[c], r)
        cols[c][idx_r:idx_r+1] = []

    def process_query(q):
        r, c = q
        if wall_exists[r][c]:
            remove_wall(r, c)
        else:
            # Look Up
            idx_r_up = bisect_left(cols[c], r)
            # The wall above is at idx_r_up - 1
            up_wall = (cols[c][idx_r_up-1], c) if idx_r_up > 0 else None
            
            # Look Down
            down_wall = (cols[c][idx_r_up], c) if idx_r_up < len(cols[c]) else None
            
            # Look Left
            idx_c_left = bisect_left(rows[r], c)
            left_wall = (r, rows[r][idx_c_left-1]) if idx_c_left > 0 else None
            
            # Look Right
            right_wall = (r, rows[r][idx_c_left]) if idx_c_left < len(rows[r]) else None
            
            # Collect all walls to be destroyed
            targets = [w for w in [up_wall, down_wall, left_wall, right_wall] if w]
            # Use map to apply remove_wall to all targets
            # We use a list comprehension to trigger the side effect
            [remove_wall(tr, tc) for tr, tc in targets]

    # Process all queries using a list comprehension
    [process_query(q) for q in queries]
    
    # Count remaining walls
    # Flatten the wall_exists grid and sum the True values
    total_remaining = sum([sum(row[1:]) for row in wall_exists])
    print(total_remaining)

if __name__ == "__main__":
    solve()