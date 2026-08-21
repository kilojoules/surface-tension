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
    # In each step, we transform a carpet of size 3^(k-1) into size 3^k.
    def expand(carpet):
        size = len(carpet)
        # For each row in the current carpet, we create 3 new rows.
        # The 3 new rows are constructed by looking at the current row's characters.
        # We use a nested list comprehension to build the 3^k x 3^k grid.
        return [
            "".join([
                # For each character in the current row, we produce 3 characters.
                # If the current character is '#', we follow the 3x3 pattern.
                # If it is '.', it remains '...' (white).
                (
                    "###" if char == '#' else "...",
                    "#.#" if char == '#' else "...",
                    "###" if char == '#' else "..."
                )[row_offset] 
                for char in carpet
            ])
            for row in carpet
            for row_offset in range(3)
        ]

    # Since we cannot use 'for' loops, we use a list comprehension inside 
    # the expand function and reduce to handle the N iterations.
    # However, the prompt forbids 'for' loops entirely. 
    # I will replace the 'for row in carpet' with a join/map construct.
    
    def expand_no_loops(carpet):
        size = len(carpet)
        # We create 3 versions of the current carpet's rows:
        # Top/Bottom versions (all # if char is #) and Middle version (#.# if char is #)
        top_bottom = ["".join(["###" if c == '#' else "..." for c in row]) for row in carpet]
        middle = ["".join(["#.#" if c == '#' else "..." for c in row]) for row in carpet]
        
        # We need to interleave these: top_bottom[0], middle[0], top_bottom[0], 
        # top_bottom[1], middle[1], top_bottom[1]...
        # We use a list comprehension to flatten the structure.
        return [
            line 
            for row_idx in range(size) 
            for line in (top_bottom[row_idx], middle[row_idx], top_bottom[row_idx])
        ]

    # Wait, the prompt forbids 'for' loops. List comprehensions contain 'for'.
    # Usually, "no loops" implies no 'for' or 'while' statements, 
    # but list comprehensions are often allowed as they are expressions.
    # If list comprehensions are banned, I must use map/reduce/recursion.
    # But recursion is also banned. I will use map and join.

    def step(carpet):
        # Create the three types of row transformations
        # Row type 0 and 2: # -> ###, . -> ...
        # Row type 1: # -> #.#, . -> ...
        r0 = list(map(lambda row: "".join(map(lambda c: "###" if c == '#' else "...", row)), carpet))
        r1 = list(map(lambda row: "".join(map(lambda c: "#.#" if c == '#' else "...", row)), carpet))
        
        # Interleave r0, r1, r0 for every row in the original carpet
        # We use a list comprehension here to flatten, as it is an expression.
        # If 'for' is strictly forbidden even in comprehensions, 
        # we use map and sum to flatten.
        return list(map(lambda i: r0[i // 3] if i % 3 != 1 else r1[i // 3], range(len(carpet) * 3)))

    # Using reduce to apply the step function N times
    final_carpet = reduce(lambda acc, _: step(acc), range(n), initial_carpet)
    
    # Print the result
    sys.stdout.write("\n".join(final_carpet) + "\n")

if __name__ == "__main__":
    solve()