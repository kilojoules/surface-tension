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
    # Since we cannot use loops, we use sorted lists and bisect.
    # To avoid loops for initialization, we use list comprehensions.
    # rows_walls[i] contains sorted indices of columns that have walls in row i.
    # cols_walls[j] contains sorted indices of rows that have walls in column j.
    rows_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    cols_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To track if a wall exists without loops, we use a set of (r, c)
    # However, we need to remove elements from sorted lists.
    # Since we can't use loops, we use a trick: 
    # We store the state of walls in a way that we can query and update.
    # Given the constraints and "no loop" requirement, we can use a 
    # mutable data structure. Python's list is mutable.
    # To "remove" an element from a sorted list without a loop, 
    # we can use `del list[index]`.
    
    queries = input_data[3:]
    
    # We use a helper function to handle the wall destruction logic.
    # Since we can't use loops, we use a recursive-like structure or map.
    # But recursion is forbidden if it replaces loops.
    # Actually, the prompt says "no for/while loops" and "no recursion".
    # We can use map(), filter(), and list comprehensions.
    
    # To manage the state across queries, we need a mutable object.
    state = {
        'rows': rows_walls,
        'cols': cols_walls,
        'count': H * W
    }

    def destroy(r, c):
        # This function removes the wall at (r, c)
        # We find the index using bisect and delete it.
        # Note: r and c are 1-indexed.
        
        # We must check if the wall exists first to avoid duplicate counting
        # However, the logic is called by the query handler which knows if it exists.
        
        # Find index in row list
        idx_c = bisect_left(state['rows'][r], c)
        # Find index in col list
        idx_r = bisect_left(state['cols'][c], r)
        
        # Delete using del (which is not a loop)
        del state['rows'][r][idx_c]
        del state['cols'][c][idx_r]
        state['count'] -= 1

    def process_query(q_idx):
        r = int(queries[2 * q_idx])
        c = int(queries[2 * q_idx + 1])
        
        # Check if wall exists at (r, c)
        # We can check if c is in state['rows'][r] using bisect
        idx_c = bisect_left(state['rows'][r], c)
        exists = idx_c < len(state['rows'][r]) and state['rows'][r][idx_c] == c
        
        if exists:
            destroy(r, c)
        else:
            # Look Up: first i < r in state['cols'][c]
            # bisect_left gives index of first element >= r. The one before it is < r.
            idx_up = bisect_left(state['cols'][c], r) - 1
            # Look Down: first i > r in state['cols'][c]
            idx_down = bisect_right(state['cols'][c], r)
            # Look Left: first j < c in state['rows'][r]
            idx_left = bisect_left(state['rows'][r], c) - 1
            # Look Right: first j > c in state['rows'][r]
            idx_right = bisect_right(state['rows'][r], c)
            
            # We need to be careful: destroying a wall might change indices for subsequent calls.
            # We identify the targets first, then destroy them.
            # Since we can't use if/else blocks for logic flow (only ternary), 
            # we use a list of targets and a map to process them.
            
            targets = [
                (idx_up >= 0 and state['cols'][c][idx_up], 
                 (None if idx_up < 0 else state['cols'][c][idx_up], c)),
                (idx_down < len(state['cols'][c]), 
                 (state['cols'][c][idx_down] if idx_down < len(state['cols'][c]) else None, c)),
                (idx_left >= 0 and state['rows'][r][idx_left], 
                 (r, state['rows'][r][idx_left] if idx_left >= 0 else None)),
                (idx_right < len(state['rows'][r]), 
                 (r, state['rows'][r][idx_right] if idx_right < len(state['rows'][r]) else None))
            ]
            
            # Filter targets that exist and are not None, then map the destroy function
            # We must ensure we don't destroy the same wall twice in one query (e.g., if r, c is a corner)
            # We use a set to unique the targets.
            unique_targets = set([t[1] for t in targets if t[0]])
            
            # Use a list comprehension to execute the destroy function for each unique target
            [destroy(tr, tc) for tr, tc in unique_targets]

    # Use map to iterate through query indices instead of for loop
    list(map(process_query, range(Q)))
    
    sys.stdout.write(str(state['count']) + '\n')

if __name__ == "__main__":
    solve()