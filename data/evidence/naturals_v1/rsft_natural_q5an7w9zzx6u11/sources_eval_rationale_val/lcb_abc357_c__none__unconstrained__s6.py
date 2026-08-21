import sys

def solve():
    # Read N from standard input
    line = sys.stdin.read().strip()
    if not line:
        return
    n = int(line)
    
    # The size of the grid is 3^N
    size = 3**n
    
    # A cell (r, c) is white if for any k in [0, n-1], 
    # the k-th digit of r and c in base 3 are both 1.
    # Otherwise, it is black.
    # We use a list comprehension to iterate through all rows and columns.
    # We use 'any' with a generator expression to check the base-3 condition.
    
    # We define a helper logic inside the comprehension:
    # (r // (3**k)) % 3 == 1 and (c // (3**k)) % 3 == 1
    
    grid = [
        "".join(
            '.' if any((r // (3**k)) % 3 == 1 and (c // (3**k)) % 3 == 1 for k in range(n))
            else '#'
            for c in range(size)
        )
        for r in range(size)
    ]
    
    # Print the resulting grid joined by newlines
    sys.stdout.write("\n".join(grid) + "\n")

if __name__ == "__main__":
    solve()