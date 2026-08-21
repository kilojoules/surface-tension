import sys

def solve():
    # Read N from standard input
    line = sys.stdin.read().strip()
    if not line:
        return
    n = int(line)
    
    # The size of the grid is 3^N
    size = 3**n
    
    # A cell (r, c) is white if there exists some k < N such that
    # (r // 3^k) % 3 == 1 AND (c // 3^k) % 3 == 1.
    # Otherwise, the cell is black.
    # We use a list comprehension to generate the grid and 
    # any() within a list comprehension to check the condition for all k.
    
    grid = [
        "".join(
            "#" if not any(
                (r // (3**k) % 3 == 1) and (c // (3**k) % 3 == 1)
                for k in range(n)
            ) else "."
            for c in range(size)
        )
        for r in range(size)
    ]
    
    # Print the resulting grid joined by newlines
    sys.stdout.write("\n".join(grid) + "\n")

if __name__ == "__main__":
    solve()