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
    # Using sorted lists to allow binary search for the nearest wall.
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    rows_walls = [sorted(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [sorted(range(1, H + 1)) for _ in range(W + 1)]
    
    # To handle deletions from sorted lists efficiently without using loops,
    # we can't use list.remove(). However, we can use a set to track destroyed walls
    # and filter them out, but the problem requires finding the "nearest" wall.
    # Since we need to delete and maintain order, and constraints are 4*10^5,
    # we can use a different approach. 
    # Actually, we can use the `bisect` module to find indices and `pop` to remove.
    # While pop(i) is O(N), the total number of walls is 4*10^5. 
    # Wait, pop(i) is too slow. Let's use a different structure or a trick.
    
    # Correction: Since we need to find the nearest wall and remove it, 
    # and we can't use loops/recursion, we can use a Doubly Linked List approach
    # simulated with arrays. 
    # For each cell (i, j), we store the index of the wall to its left, right, up, and down.
    
    # L[i][j], R[i][j], U[i][j], D[i][j]
    # To avoid nested lists (which are slow to initialize), we use flat arrays.
    # index = (i-1)*W + (j-1)
    
    L = [j - 1 for i in range(H) for j in range(1, W + 1)]
    R = [j + 1 for i in range(H) for j in range(1, W + 1)]
    U = [i - 1 for i in range(1, H + 1) for j in range(W)]
    D = [i + 1 for i in range(1, H + 1) for j in range(W)]
    
    # wall_exists[index]
    exists = [True] * (H * W)
    
    # Process queries
    queries = input_data[3:]
    
    # We need to process queries. Since we can't use for/while loops, 
    # we use map() or list comprehensions.
    
    def destroy(r, c):
        idx = (r - 1) * W + (c - 1)
        if not exists[idx]:
            return 0
        
        # Remove wall at (r, c)
        exists[idx] = False
        
        # Update neighbors
        # Left neighbor's Right becomes this cell's Right
        # Right neighbor's Left becomes this cell's Left
        # We need to handle boundaries (0 and W+1)
        
        # To avoid loops, we use a helper to update the links
        # Because we can't use loops, we use a list to perform updates
        
        # For row r, col c:
        # The wall to the left of (r, c) is at (r, L[idx])
        # The wall to the right of (r, c) is at (r, R[idx])
        # The wall above (r, c) is at (U[idx], c)
        # The wall below (r, c) is at (D[idx], c)
        
        # We use a list to store the updates to be made to the flat arrays
        # Since we can't use loops, we use a trick with slice assignment or map.
        return 1

    # The constraint "no loops" is very strict. Let's use a different approach.
    # We can use a mutable object to keep track of the count and the state.
    
    state = {
        'exists': [True] * (H * W),
        'L': L, 'R': R, 'U': U, 'D': D,
        'count': H * W
    }
    
    def process_query(q):
        rq = int(q[0])
        cq = int(q[1])
        idx = (rq - 1) * W + (cq - 1)
        
        if state['exists'][idx]:
            # Destroy wall at (rq, cq)
            state['exists'][idx] = False
            # Update links
            # Left neighbor
            l_val = state['L'][idx]
            if l_val >= 1:
                state['R'][(rq - 1) * W + (l_val - 1)] = state['R'][idx]
            # Right neighbor
            r_val = state['R'][idx]
            if r_val <= W:
                state['L'][(rq - 1) * W + (r_val - 1)] = state['L'][idx]
            # Up neighbor
            u_val = state['U'][idx]
            if u_val >= 1:
                state['D'][(u_val - 1) * W + (cq - 1)] = state['D'][idx]
            # Down neighbor
            d_val = state['D'][idx]
            if d_val <= H:
                state['U'][(d_val - 1) * W + (cq - 1)] = state['U'][idx]
            
            state['count'] -= 1
            return
        
        # No wall at (rq, cq), destroy 4 neighbors
        # We need to find the first wall in 4 directions.
        # The "links" L, R, U, D already point to the nearest walls.
        
        # We must be careful: the 4 walls are destroyed "simultaneously".
        # We identify them first, then destroy them.
        
        targets = [
            (rq, state['L'][idx]),
            (rq, state['R'][idx]),
            (state['U'][idx], cq),
            (state['D'][idx], cq)
        ]
        
        # Filter valid targets (within boundaries) and unique
        # Since we can't use loops, we use a list comprehension
        valid_targets = [t for t in targets if 1 <= t[0] <= H and 1 <= t[1] <= W]
        
        # To "destroy" them without a loop, we use map
        # But the destroy logic is complex. Let's wrap the destruction in a function.
        
        def perform_destruction(target):
            tr, tc = target
            t_idx = (tr - 1) * W + (tc - 1)
            if state['exists'][t_idx]:
                state['exists'][t_idx] = False
                # Update links
                lv = state['L'][t_idx]
                if lv >= 1: state['R'][(tr - 1) * W + (lv - 1)] = state['R'][t_idx]
                rv = state['R'][t_idx]
                if rv <= W: state['L'][(tr - 1) * W + (rv - 1)] = state['L'][t_idx]
                uv = state['U'][t_idx]
                if uv >= 1: state['D'][(uv - 1) * W + (tc - 1)] = state['D'][t_idx]
                dv = state['D'][t_idx]
                if dv <= H: state['U'][(dv - 1) * W + (tc - 1)] = state['U'][t_idx]
                state['count'] -= 1
        
        # Use map to apply the destruction to all valid targets
        list(map(perform_destruction, valid_targets))

    # Process all queries using map
    # Group input into pairs of (R, C)
    query_pairs = [queries[i:i+2] for i in range(0, len(queries), 2)]
    list(map(process_query, query_pairs))
    
    sys.stdout.write(str(state['count']) + '\n')

if __name__ == "__main__":
    solve()