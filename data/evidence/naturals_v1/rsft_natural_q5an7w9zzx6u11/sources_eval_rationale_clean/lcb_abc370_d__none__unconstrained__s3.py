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
    queries = input_data[3:]

    # We need to maintain sorted lists of wall indices for every row and column.
    # rows[i] contains columns j where a wall exists at (i, j).
    # cols[j] contains rows i where a wall exists at (i, j).
    # Using lists and bisect for O(log N) lookup and O(N) deletion.
    # Given H*W <= 4e5, O(N) deletion in a list might be slow, 
    # but Python's list.pop() is highly optimized.
    
    # Initialize rows and cols
    # Using 1-based indexing for convenience with the problem statement
    rows = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To track if a wall exists at (r, c) without scanning lists
    # We use a set of destroyed walls for O(1) check
    # However, the problem asks for remaining walls, so we track destroyed ones.
    destroyed = set()

    def remove_wall(r, c, state):
        # state is (rows, cols, destroyed)
        rs, cs, dst = state
        if (r, c) in dst:
            return state
        
        # Find index in row list and column list
        # Since we can't use loops, we use bisect to find the exact index
        r_idx = bisect_left(rs[r], c)
        c_idx = bisect_left(cs[c], r)
        
        # Remove from lists
        # Note: list.pop(i) is used. 
        # We must create new references or mutate in place. 
        # reduce allows mutation of the objects passed in the accumulator.
        rs[r].pop(r_idx)
        cs[c].pop(c_idx)
        dst.add((r, c))
        return state

    def process_query(state, q_text):
        rs, cs, dst = state
        r, c = map(int, q_text.split(','))
        
        if (r, c) not in dst:
            # Wall exists at (r, c), destroy it and stop
            return remove_wall(r, c, state)
        
        # No wall at (r, c), destroy neighbors
        # 1. Up (same column, smaller row)
        c_list = cs[c]
        idx = bisect_left(c_list, r)
        if idx > 0:
            remove_wall(c_list[idx-1], c, state)
            
        # 2. Down (same column, larger row)
        if idx < len(c_list):
            remove_wall(c_list[idx], c, state)
            
        # 3. Left (same row, smaller column)
        r_list = rs[r]
        idx_r = bisect_left(r_list, c)
        if idx_r > 0:
            remove_wall(r, r_list[idx_r-1], state)
            
        # 4. Right (same row, larger column)
        if idx_r < len(r_list):
            remove_wall(r, r_list[idx_r], state)
            
        return state

    # Prepare queries as comma-separated strings to pass into reduce
    # since reduce takes two arguments (accumulator, current_item)
    query_pairs = [f"{queries[i*2]},{queries[i*2+1]}" for i in range(Q)]
    
    # Initial state: (rows, cols, destroyed_set)
    initial_state = (rows, cols, destroyed)
    
    # Process all queries using reduce
    final_state = reduce(process_query, query_pairs, initial_state)
    
    # The number of remaining walls is total - size of destroyed set
    print(H * W - len(final_state[2]))

if __name__ == "__main__":
    solve()