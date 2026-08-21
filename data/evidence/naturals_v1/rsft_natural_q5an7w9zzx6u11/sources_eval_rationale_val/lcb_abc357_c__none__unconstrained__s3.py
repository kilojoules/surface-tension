import sys
from functools import reduce

def solve():
    # Read N from standard input
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())

    # A level-0 carpet is just a 1x1 grid with a black cell.
    # We use reduce to evolve the carpet from level 0 up to level N.
    # The state being passed through reduce is the grid (list of strings).
    
    def expand(carpet):
        # Current size of the carpet
        size = len(carpet)
        
        # For each row in the 3x3 block structure:
        # Row group 0: Top 3 blocks
        # Row group 1: Middle 3 blocks (where the center is white)
        # Row group 2: Bottom 3 blocks
        
        # We construct the new grid by iterating through the indices of the 3x3 macro-grid.
        # We use a nested list comprehension to build the rows.
        return [
            # For each row 'r' in the 3^K grid:
            # Determine which macro-row (0, 1, or 2) it belongs to.
            # Determine which micro-row within the level-(K-1) carpet it corresponds to.
            # Then construct the string by concatenating 3 macro-columns.
            
            # The logic for the character at (r, c):
            # It is white ('.') if the current macro-row is 1 AND macro-col is 1.
            # Otherwise, it is the character from the level-(K-1) carpet at (r % size, c % size).
            
            "".join([
                # Macro-column j (0, 1, 2)
                # If macro-row is 1 and macro-col is 1, it's a white block.
                # Otherwise, it's the corresponding character from the previous carpet.
                (
                    "." if (r // size == 1 and j == 1) 
                    else carpet[r % size][c % size]
                )
                for j in range(3) 
                for c in range(size)
            ])
            for r in range(3 * size)
        ]

    # Start with level 0: ["#"]
    # Apply the expand function N times.
    final_carpet = reduce(lambda acc, _: expand(acc), range(n), ["#"])
    
    # Print the resulting grid
    sys.stdout.write("\n".join(final_carpet) + "\n")

if __name__ == "__main__":
    solve()