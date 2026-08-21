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
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # We maintain the state as a set of destroyed walls.
    # To efficiently find the "nearest" wall, we need to know which walls still exist.
    # However, since we cannot use loops, we will store the set of destroyed walls
    # and use list comprehensions to filter and find the min/max indices.
    # Given the constraints (H*W <= 4e5), we must be careful with complexity.
    # Actually, the most efficient way to find the nearest wall without loops 
    # is to maintain sorted lists of existing walls for each row and column.
    
    # Since we cannot use loops, we use a dictionary of sets to track existing walls.
    # rows[r] = set of columns that have a wall in row r
    # cols[c] = set of rows that have a wall in column c
    initial_state = {
        'rows': {r: set(range(1, W + 1)) for r in range(1, H + 1)},
        'cols': {c: set(range(1, H + 1)) for c in range(1, W + 1)},
        'count': H * W
    }

    def process_query(state, query):
        r, c = query
        rows, cols = state['rows'], state['cols']
        
        # Check if wall exists at (r, c)
        if c in rows[r]:
            # Destroy wall at (r, c)
            rows[r].remove(c)
            cols[c].remove(r)
            return {**state, 'rows': rows, 'cols': cols, 'count': state['count'] - 1}
        
        # No wall at (r, c), find 4 nearest walls
        # We use list comprehensions to find the targets
        # Up: max i < r where (i, c) has a wall
        # Down: min i > r where (i, c) has a wall
        # Left: max j < c where (r, j) has a wall
        # Right: min j > c where (r, j) has a wall
        
        targets = [
            # Up
            (max([i for i in cols[c] if i < r], default=None), c),
            # Down
            (min([i for i in cols[c] if i > r], default=None), c),
            # Left
            (r, max([j for j in rows[r] if j < c], default=None)),
            # Right
            (r, min([j for j in rows[r] if j > c], default=None))
        ]
        
        # Filter out None values
        valid_targets = [t for t in targets if None not in t]
        
        # To avoid loops, we use a helper function to remove walls
        def remove_wall(s, target):
            tr, tc = target
            if tc in s['rows'][tr]:
                s['rows'][tr].remove(tc)
                s['cols'][tc].remove(tr)
                return {**s, 'count': s['count'] - 1}
            return s

        # Use reduce to apply the removal of the 4 potential walls
        return reduce(remove_wall, valid_targets, state)

    final_state = reduce(process_query, queries, initial_state)
    print(final_state['count'])

if __name__ == "__main__":
    solve()