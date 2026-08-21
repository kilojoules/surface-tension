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
    
    # Queries as a list of (R, C) tuples
    queries = [ (int(input_data[i]), int(input_data[i+1])) 
                for i in range(3, len(input_data), 2) ]

    # We use a set to store destroyed walls. 
    # reduce will act as our "loop" over the queries.
    # The accumulator is the set of destroyed walls.
    final_destroyed = reduce(
        lambda destroyed, q: (
            # If wall exists at (R, C), destroy it
            destroyed | {(q[0], q[1])} if (q[0], q[1]) not in destroyed else (
                # Otherwise, find the first wall in 4 directions
                # We use generator expressions inside next() to find the first 
                # coordinate NOT in the destroyed set.
                destroyed | {
                    # Up
                    next(( (i, q[1]) for i in range(q[0]-1, 0, -1) 
                          if (i, q[1]) not in destroyed ), None),
                    # Down
                    next(( (i, q[1]) for i in range(q[0]+1, H+1) 
                          if (i, q[1]) not in destroyed ), None),
                    # Left
                    next(( (q[0], j) for j in range(q[1]-1, 0, -1) 
                          if (q[0], j) not in destroyed ), None),
                    # Right
                    next(( (q[0], j) for j in range(q[1]+1, W+1) 
                          if (q[0], j) not in destroyed ), None)
                }
            )
        ),
        set(),
        queries
    )

    # The number of remaining walls is Total - Destroyed.
    # We filter None values that might have been added to the set.
    actual_destroyed_count = len([w for w in final_destroyed if w is not None])
    sys.stdout.write(str(H * W - actual_destroyed_count) + '\n')

if __name__ == "__main__":
    solve()