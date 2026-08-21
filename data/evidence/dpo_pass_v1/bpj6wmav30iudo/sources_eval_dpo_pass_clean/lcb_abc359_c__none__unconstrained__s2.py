import sys

def solve():
    # Read input from standard input
    # Using map and split to handle the input without explicit loops
    input_data = sys.stdin.read().split()
    
    # Assign coordinates
    # Sx, Sy: Starting point
    # Tx, Ty: Target point
    Sx, Sy, Tx, Ty = map(int, input_data)

    # The tiling pattern is as follows:
    # If i + j is even, (i, j) and (i+1, j) are in the same tile.
    # This means tiles are 2x1 horizontal blocks starting at (i, j) where i+j is even.
    # Let's transform the coordinates to a system where each unit is a tile.
    # For a fixed j, the tiles are {(0,j), (1,j)}, {(2,j), (3,j)}... if j is even.
    # If j is odd, the tiles are {(-1,j), (0,j)}, {(1,j), (2,j)}...
    
    # A point (x, y) is in tile (X, Y) where:
    # Y = y
    # X = (x if y % 2 == 0 else x + 1) // 2
    # However, the rule is i+j is even.
    # If j is even, i must be even for the tile to start at (i, j).
    # If j is odd, i must be odd for the tile to start at (i, j).
    
    # Let's define the tile coordinates (U, V):
    # V = y
    # U = (x - (1 if (y % 2 != 0) else 0)) // 2
    # Wait, a simpler way:
    # The tile containing (x, y) is identified by ( (x + (y % 2)) // 2, y )
    
    # Let Ux = (Sx + (Sy % 2)) // 2
    # Let Uy = Sy
    # Let Vx = (Tx + (Ty % 2)) // 2
    # Let Vy = Ty
    
    # The cost to move between tiles:
    # Moving vertically (Uy to Vy) always costs 1 per unit.
    # Moving horizontally (Ux to Vx) costs 1 per unit.
    # However, we can move diagonally in the tile grid.
    # One step in V (vertical) and one step in U (horizontal) 
    # can sometimes be optimized.
    
    # Let dx = abs(Ux - Vx)
    # Let dy = abs(Uy - Vy)
    # The minimum toll is max(dx, dy)
    
    # Let's verify with Sample 1: 5 0 -> 2 5
    # S: Ux = (5 + 0)//2 = 2, Uy = 0
    # T: Vx = (2 + (5%2))//2 = (2+1)//2 = 1, Vy = 5
    # dx = |2 - 1| = 1, dy = |0 - 5| = 5. max(1, 5) = 5. Correct.
    
    # Sample 2: 3 1 -> 4 1
    # S: Ux = (3 + 1)//2 = 2, Uy = 1
    # T: Vx = (4 + 1)//2 = 2, Vy = 1
    # dx = 0, dy = 0. max(0, 0) = 0. Correct.
    
    # Calculation using the derived formula
    # We use a list comprehension or direct assignment to avoid loops
    res = max(abs(((Sx + (Sy % 2)) // 2) - ((Tx + (Ty % 2)) // 2)), abs(Sy - Ty))
    
    # Print the result
    sys.stdout.write(str(res) + '\n')

if __name__ == "__main__":
    solve()