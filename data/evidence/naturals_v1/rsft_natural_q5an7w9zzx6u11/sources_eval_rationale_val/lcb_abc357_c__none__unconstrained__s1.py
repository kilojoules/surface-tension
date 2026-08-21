import sys
from functools import reduce

def solve():
    # Read N from standard input
    line = sys.stdin.read().strip()
    if not line:
        return
    n = int(line)

    # A level-0 carpet is a 1x1 grid with a single black cell '#'
    initial_carpet = ["#"]

    # The transformation function for reduce:
    # It takes the carpet of level K-1 and produces the carpet of level K.
    # For each row in the previous carpet, we create 3 new rows.
    # For each new row, we determine if it's a 'border' row (top/bottom) 
    # or a 'middle' row of the 3x3 block structure.
    def expand(carpet):
        size = len(carpet)
        # We create a range of 3*size indices to build the new grid
        return [
            "".join([
                # Determine the character for column j
                # The cell is white if:
                # 1. The current block is the central block (size <= j < 2*size)
                #    AND the current row is in the central block (size <= i < 2*size)
                # 2. Otherwise, it depends on the value in the level-(K-1) carpet
                #    at position (i % size, j % size)
                "." if (size <= i // 1 < 2 * size and size <= j // 1 < 2 * size) # This logic is slightly wrong for indices
                else carpet[i % size][j % size]
                for j in range(3 * size)
            ])
            for i in range(3 * size)
        ]
    
    # Corrected logic inside a list comprehension:
    # For a cell (i, j) in level K, it is white if:
    # (i // size == 1) AND (j // size == 1)
    # Otherwise, it inherits the color from level K-1 at (i % size, j % size)
    
    def step(carpet):
        s = len(carpet)
        return [
            "".join([
                "." if (i // s == 1 and j // s == 1) else carpet[i % s][j % s]
                for j in range(3 * s)
            ])
            for i in range(3 * s)
        ]

    # Use reduce to apply the step function N times
    result = reduce(lambda acc, _: step(acc), range(n), initial_carpet)
    
    # Print the final grid
    sys.stdout.write("\n".join(result) + "\n")

if __name__ == "__main__":
    solve()