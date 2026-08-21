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

    # We use reduce to evolve the carpet from level 0 up to level N.
    # In each step, we take the current carpet (size 3^k) and 
    # construct the next carpet (size 3^{k+1}).
    def expand(carpet):
        size = len(carpet)
        # For each row in the 3x3 block structure:
        # Row group 1: Top 3 blocks (carpet, carpet, carpet)
        # Row group 2: Middle 3 blocks (carpet, white, carpet)
        # Row group 3: Bottom 3 blocks (carpet, carpet, carpet)
        
        # We create a helper to generate the rows for the current expansion level.
        # The 'i' represents the index of the row within the previous carpet.
        # The 'r' represents which of the 3 vertical sections we are in (0, 1, 2).
        
        # To avoid loops, we use a nested list comprehension.
        # We iterate r in (0, 1, 2) and i in range(size).
        return [
            # For each row i of the previous carpet:
            # If r == 1 (middle section), the middle block is white ('.'*size)
            # Otherwise, all three blocks are the row carpet[i]
            (carpet[i] + carpet[i] + carpet[i]) if r != 1 else 
            (carpet[i] + "." * size + carpet[i])
            for r in range(3) 
            for i in range(size)
        ]

    # Apply the expand function N times starting from the level-0 carpet.
    final_carpet = reduce(lambda acc, _: expand(acc), range(n), initial_carpet)

    # Print the resulting grid joined by newlines.
    sys.stdout.write("\n".join(final_carpet) + "\n")

if __name__ == "__main__":
    solve()