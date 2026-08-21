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
    # In each step, we transform a carpet of size 3^K into size 3^(K+1).
    def expand(carpet):
        size = len(carpet)
        # For each row in the current carpet, we create 3 new rows.
        # For each row, we create 3 versions of the row string based on the 3x3 block rule.
        # The 3-row pattern for a block is:
        # Row 1: Top (Left, Mid, Right)
        # Row 2: Mid (Left, Center(white), Right)
        # Row 3: Bot (Left, Mid, Right)
        
        # We generate the rows of the new carpet by iterating through the old rows
        # and applying the 3x3 expansion logic.
        return [
            row 
            for r in carpet 
            for row in [
                # Top row of the 3x3 block: Left, Mid, Right
                "".join([r, r, r]),
                # Middle row: Left, Center (white), Right
                "".join([r, "." * size, r]),
                # Bottom row: Left, Mid, Right
                "".join([r, r, r])
            ]
            # However, the above logic is slightly wrong because the 'Center' 
            # white space must be applied to the entire middle block of the 3x3.
            # Let's refine: for a row 'r' in the current carpet, 
            # it contributes to 3 rows in the new carpet.
            # But the middle row of the 3x3 grid is special.
        ]

    # Corrected expand logic using list comprehensions:
    # For each row 'r' in the current carpet:
    # The new carpet will have 3*size rows.
    # We can think of this as: 
    # For i in 0..size-1:
    #   NewRow(3*i)   = r + r + r
    #   NewRow(3*i+1) = r + (r if not in middle_block else '.') + r
    #   NewRow(3*i+2) = r + r + r
    # Wait, the middle block is white ONLY if the current coordinate is in the center.
    # The recursive definition says: the central 3^(K-1) block is white.
    # That means for rows in the middle third of the 3^K grid, 
    # the middle third of the columns are white.
    
    def grow(carpet):
        size = len(carpet)
        # Create the 3 versions of the row strings
        # Type A: r + r + r (for top and bottom thirds of the 3x3)
        # Type B: r + r + r (for top/bottom thirds of the middle 3x3)
        # Type C: r + "."*size + r (for the middle third of the middle 3x3)
        
        # We need to construct the rows such that:
        # Rows 0 to size-1: r + r + r
        # Rows size to 2*size-1: r + ('.' if row is in middle else r) + r
        # Rows 2*size to 3*size-1: r + r + r
        
        # Let's use a comprehension that references the index to handle the middle block.
        return [
            "".join([
                carpet[i % size], 
                ("." * size) if (i // size == 1) else carpet[i % size], 
                carpet[i % size]
            ])
            for i in range(3 * size)
        ]

    # Use reduce to apply 'grow' N times
    final_carpet = reduce(lambda acc, _: grow(acc), range(n), initial_carpet)
    
    # Print the result
    sys.stdout.write("\n".join(final_carpet) + "\n")

if __name__ == "__main__":
    solve()