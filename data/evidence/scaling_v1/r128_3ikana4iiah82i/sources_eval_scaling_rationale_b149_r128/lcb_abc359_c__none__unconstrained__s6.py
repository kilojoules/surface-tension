import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1. 
    # When i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3), (4,5)... if j is even
    # and (-1,0), (1,2), (3,4)... if j is odd.
    
    # Let's define a coordinate transformation to a grid of tiles.
    # A tile can be identified by (column, row).
    # For a square (i, j), the row is j.
    # The column depends on the parity of i+j.
    # If i+j is even, square (i, j) and (i+1, j) are one tile.
    # We can map i to a tile index:
    # If j is even: i=0,1 -> col 0; i=2,3 -> col 1... (i // 2)
    # If j is odd: i=-1,0 -> col 0; i=1,2 -> col 1... ((i+1) // 2)
    
    # More simply: the tile index for i at height j is (i + (j % 2)) // 2.
    
    s_col = (sx + (sy % 2)) // 2
    s_row = sy
    t_col = (tx + (ty % 2)) // 2
    t_row = ty
    
    # The cost to move between tiles:
    # Moving vertically (changing row) always enters a new tile.
    # Moving horizontally (changing col) might enter a new tile.
    # However, the problem says we can move n units in one direction.
    # If we move vertically from sy to ty, we pass through abs(ty - sy) tiles.
    # If we move horizontally from sx to tx, we might stay in the same tile or move across.
    
    # Let's analyze the cost:
    # To get from (sx, sy) to (tx, ty):
    # 1. Vertical distance: dy = abs(sy - ty). Each step is a new tile.
    # 2. Horizontal distance: dx = abs(s_col - t_col).
    
    # The total cost is dy + dx, but we must check if the starting and ending
    # tiles are the same. If we are already in the target tile, the cost is 0.
    # But the move rules say "Each time he enters a tile, he pays a toll of 1."
    # This implies the starting tile is free.
    
    # The distance in the transformed grid is the Manhattan distance.
    # Cost = abs(s_row - t_row) + abs(s_col - t_col)
    
    print(abs(s_row - t_row) + abs(s_col - t_col))

if __name__ == "__main__":
    solve()