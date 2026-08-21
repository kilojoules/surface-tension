import sys

def solve():
    # Read input from stdin
    # Using map and split to handle the input without explicit loops
    input_data = sys.stdin.read().split()
    
    # Assign coordinates
    # Sx, Sy: Start point; Tx, Ty: Target point
    # We use a list comprehension to convert strings to ints
    coords = list(map(int, input_data))
    Sx, Sy, Tx, Ty = coords[0], coords[1], coords[2], coords[3]

    # The tiling pattern:
    # If i+j is even, (i, j) and (i+1, j) are one tile.
    # This means tiles are 2x1 horizontal blocks.
    # In row j, if j is even, tiles are [0,1], [2,3]... (i=0,2,4...)
    # In row j, if j is odd, tiles are [-1,0], [1,2]... (i=1,3,5...)
    
    # Let's transform the coordinates to a grid where each cell is a tile.
    # For a fixed y, the x-boundary depends on whether y is even or odd.
    # If y is even, x is grouped as (0,1), (2,3)... -> x // 2
    # If y is odd, x is grouped as (1,2), (3,4)... -> (x-1) // 2
    
    # However, a simpler way to view this is:
    # Moving vertically always crosses a tile boundary (toll 1).
    # Moving horizontally crosses a boundary every 2 units.
    # The "cost" to move from (Sx, Sy) to (Tx, Ty) is:
    # Vertical distance: abs(Sy - Ty)
    # Horizontal distance: The number of boundaries crossed.
    # In row y, boundaries are at x where (x + y) is even.
    
    # Let's define the "tile index" for a point (x, y):
    # The tile containing (x, y) is identified by (y, (x + (y % 2)) // 2)
    
    # Let Ux, Uy be the tile coordinates of (Sx, Sy)
    # Let Vx, Vy be the tile coordinates of (Tx, Ty)
    Uy = Sy
    Ux = (Sx + (Sy % 2)) // 2
    
    Vy = Ty
    Vx = (Tx + (Ty % 2)) // 2
    
    # The distance between tiles in this transformed grid is the Manhattan distance.
    # Each step in the transformed grid costs 1 toll.
    # Distance = abs(Ux - Vx) + abs(Uy - Vy)
    
    # We use a formatted string to output the result.
    sys.stdout.write(str(abs(Ux - Vx) + abs(Uy - Vy)) + '\n')

# Execute the solver
if __name__ == "__main__":
    solve()