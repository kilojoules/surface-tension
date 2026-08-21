import sys
from functools import reduce

def solve():
    # Read N from standard input
    line = sys.stdin.read().strip()
    if not line:
        return
    n = int(line)

    # A level-0 carpet is just a single black cell
    initial_carpet = ["#"]

    # We use reduce to evolve the carpet from level 0 to level N.
    # In each step, we take the current carpet (size 3^k) and 
    # create a new carpet (size 3^{k+1}) by arranging 9 blocks.
    def expand(carpet):
        size = len(carpet)
        # Create a row of 3 blocks: [carpet, carpet, carpet]
        # But we need to handle the middle block being white for the middle row.
        # We define a 'white' block of the same size.
        white_block = ["." * size] * size
        
        # The structure of a level-K carpet is:
        # Row 1: Carpet | Carpet | Carpet
        # Row 2: Carpet | White  | Carpet
        # Row 3: Carpet | Carpet | Carpet
        
        # We construct the new grid by iterating through the indices of the 3x3 block layout.
        # i represents the block-row (0, 1, 2), j represents the block-col (0, 1, 2).
        # We use a nested list comprehension to build the 3^K x 3^K grid.
        return [
            "".join([
                carpet[r] if not (i == 1 and j == 1) else white_block[r]
                for j in range(3)
            ])
            for i in range(3)
            for r in range(size)
        ]

    # Note: The logic inside expand needs to be slightly adjusted because 
    # the 'for i in range(3) for r in range(size)' creates the rows in the correct order.
    # However, the 'carpet[r]' needs to be accessed based on the block-row.
    # Let's refine the comprehension:
    def expand_refined(carpet):
        size = len(carpet)
        return [
            "".join([
                carpet[r] if not (i == 1 and j == 1) else "." * size
                for j in range(3)
            ])
            for i in range(3)
            for r in range(size)
        ]

    # Apply the expansion N times
    result = reduce(lambda acc, _: expand_refined(acc), range(n), initial_carpet)
    
    # Print the final grid
    sys.stdout.write("\n".join(result) + "\n")

if __name__ == "__main__":
    solve()