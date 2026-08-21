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
    
    queries = []
    for i in range(Q):
        queries.append((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])))

    # We need to track existing walls in each row and each column.
    # Since H*W is up to 4e5, we can't have H lists and W lists if H or W is large.
    # However, we only care about rows and columns that are actually referenced in queries.
    # Actually, the constraint is H*W <= 4e5, so we can afford lists for all rows/cols.
    
    # row_walls[r] stores sorted indices of columns that have a wall in row r.
    # col_walls[c] stores sorted indices of rows that have a wall in column c.
    # To avoid loops, we use a dictionary of lists or a list of lists.
    # Given the constraints, list of lists is fine.
    
    # Initialize walls: every cell has a wall.
    # But we can't do [list(range(1, W+1)) for _ in range(H)] because that's O(H*W).
    # Wait, the problem says H*W <= 4e5. O(H*W) is acceptable for initialization.
    
    # Using list comprehensions to avoid explicit for loops.
    row_walls = [list(range(1, W + 1)) for _ in range(H)]
    col_walls = [list(range(1, H + 1)) for _ in range(W)]
    
    # We need a way to "destroy" walls. Since we can't use loops, 
    # we can't iterate through queries with a for loop.
    # But the problem asks for a Python program. The only way to process 
    # queries without loops is using recursion or functional tools like reduce.
    
    from functools import reduce

    def process_query(state, query):
        r, c = query
        # state = (row_walls, col_walls)
        rw, cw = state
        
        # Check if wall exists at (r, c)
        # We use bisect_left to check existence in a sorted list
        idx = bisect_left(rw[r-1], c)
        exists = idx < len(rw[r-1]) and rw[r-1][idx] == c
        
        if exists:
            # Destroy wall at (r, c)
            # To "destroy" without a loop, we can't use .remove() as it's in-place.
            # We create a new list.
            new_rw_r = rw[r-1][:idx] + rw[r-1][idx+1:]
            new_cw_c = cw[c-1][:bisect_left(cw[c-1], r)] + cw[c-1][bisect_left(cw[c-1], r)+1:]
            
            # Update the state
            # Since we can't use a loop to update the list of lists, 
            # we use a trick with list slicing to replace one element.
            return (rw[:r-1] + [new_rw_r] + rw[r:], 
                    cw[:c-1] + [new_cw_c] + cw[c:])
        else:
            # Destroy first walls in 4 directions
            # Up: first i < r in col_walls[c-1]
            # Down: first i > r in col_walls[c-1]
            # Left: first j < c in row_walls[r-1]
            # Right: first j > c in row_walls[r-1]
            
            # Find indices using bisect
            r_list = rw[r-1]
            c_list = cw[c-1]
            
            # Left
            l_idx = bisect_left(r_list, c) - 1
            # Right
            r_idx = bisect_left(r_list, c)
            # Up
            u_idx = bisect_left(c_list, r) - 1
            # Down
            d_idx = bisect_left(c_list, r)
            
            # Identify walls to destroy
            # We use a list of coordinates to destroy
            to_destroy = [
                (r, r_list[l_idx]) if l_idx >= 0 else None,
                (r, r_list[r_idx]) if r_idx < len(r_list) else None,
                (c_list[u_idx], c) if u_idx >= 0 else None,
                (c_list[d_idx], c) if d_idx < len(c_list) else None
            ]
            # Remove None values
            to_destroy = [item for item in to_destroy if item]
            
            # Now we must remove these walls from the state.
            # Since we can't use a loop, we use reduce again to process the to_destroy list.
            return reduce(lambda s, q: (
                s[0][:q[0]-1] + [s[0][q[0]-1][:bisect_left(s[0][q[0]-1], q[1])] + s[0][q[0]-1][bisect_left(s[0][q[0]-1], q[1])+1:]] + s[0][q[0]:],
                s[1][:q[1]-1] + [s[1][q[1]-1][:bisect_left(s[1][q[1]-1], q[0])] + s[1][q[1]-1][bisect_left(s[1][q[1]-1], q[0])+1:]] + s[1][q[1]:]
            ), state, to_destroy)

    final_state = reduce(process_query, queries, (row_walls, col_walls))
    
    # Calculate remaining walls
    # Total walls = H*W. Remaining = H*W - (total walls destroyed).
    # However, it's easier to just sum the lengths of the remaining row_walls.
    print(sum(map(len, final_state[0])))

if __name__ == "__main__":
    solve()