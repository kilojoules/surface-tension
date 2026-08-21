import sys

def solve():
    # Read the starting and target coordinates from standard input
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        sx, sy = map(int, line1)
        line2 = sys.stdin.readline().split()
        if not line2:
            return
        tx, ty = map(int, line2)
    except ValueError:
        return

    # The tiling pattern described is a "brick-like" pattern.
    # For a square A_{i,j}, it belongs to a 2x1 tile (horizontal) if:
    # i+j is even, then A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j:
    # If j is even: A_{0,j} & A_{1,j}, A_{2,j} & A_{3,j}, ... are tiles.
    # If j is odd: A_{-1,j} & A_{0,j}, A_{1,j} & A_{2,j}, ... are tiles.
    
    # Let's define a coordinate system for the tiles.
    # A tile can be identified by (tile_x, tile_y).
    # For a cell (x, y):
    # If (x + y) is even, it's the left half of a tile if we consider the pair (x,y) and (x+1,y).
    # Actually, the rule says: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This implies:
    # For a fixed y:
    # If y is even: cells (0,y)&(1,y), (2,y)&(3,y) ... are tiles.
    # If y is odd: cells (-1,y)&(0,y), (1,y)&(2,y) ... are tiles.
    
    # Let's map cell (x, y) to a tile ID (tx, ty).
    # ty = y
    # If y is even: tx = x // 2
    # If y is odd: tx = (x + 1) // 2 (since x=0 is the right half of tile -1, 1 is left of 0)
    # Wait, if y is odd, cell x=0 is paired with x=-1. So x=0 belongs to tile (0-1)//2 = -1? 
    # Let's re-evaluate:
    # If y is even: x=0,1 -> tile 0; x=2,3 -> tile 1; x=-2,-1 -> tile -1. Formula: tx = x // 2.
    # If y is odd: x=-1,0 -> tile 0; x=1,2 -> tile 1; x=3,4 -> tile 2. Formula: tx = (x + 1) // 2.
    
    def get_tile_coords(x, y):
        ty = y
        if y % 2 == 0:
            tx = x // 2
        else:
            tx = (x + 1) // 2
        return tx, ty

    sx_tile, sy_tile = get_tile_coords(sx, sy)
    tx_tile, ty_tile = get_tile_coords(tx, ty)

    # Distance in a grid where moving to an adjacent tile costs 1.
    # However, this is not a simple grid. 
    # Moving vertically from (tx, ty) to (tx, ty+1):
    # The tile at (tx, ty) covers some x-range. The tile at (tx, ty+1) covers some x-range.
    # Do they overlap?
    # If ty is even, tile tx covers [2*tx, 2*tx + 2).
    # If ty+1 is odd, tile tx covers [2*tx-1, 2*tx + 1).
    # They overlap on [2*tx, 2*tx + 1). So moving vertically is always 1 toll.
    
    # Moving horizontally from (tx, ty) to (tx+1, ty):
    # This is always 1 toll because they are distinct tiles.
    
    # The cost to move from (sx_tile, sy_tile) to (tx_tile, ty_tile) is:
    # dx = abs(sx_tile - tx_tile)
    # dy = abs(sy_tile - ty_tile)
    # The distance is dx + dy.
    
    # But wait, can we move diagonally?
    # If we move from (tx, ty) to (tx+1, ty+1), we enter two new tiles.
    # Is there a way to enter only one?
    # From tile (tx, ty), we can move to any tile that shares a boundary.
    # Tiles at (tx, ty) and (tx, ty+1) always share a boundary.
    # Tiles at (tx, ty) and (tx+1, ty) always share a boundary.
    # What about (tx, ty) and (tx+1, ty+1)?
    # If ty is even: Tile(tx, ty) is [2tx, 2tx+2] x [ty, ty+1]
    # Tile(tx+1, ty+1) is [2(tx+1)-1, 2(tx+1)+1] x [ty+1, ty+2] = [2tx+1, 2tx+3] x [ty+1, ty+2]
    # They share the boundary segment [2tx+1, 2tx+2] at y = ty+1.
    # Thus, we can move from (tx, ty) to (tx+1, ty+1) by paying only 1 toll!
    
    # This means the distance is max(dx, dy) if we can move diagonally.
    # Let's check: if we move from (tx, ty) to (tx+1, ty+1), the cost is 1.
    # This is a Chebyshev distance if diagonal moves are allowed.
    # In this specific tiling, the "diagonal" move (tx, ty) -> (tx+1, ty+1) 
    # and (tx, ty) -> (tx-1, ty+1) are possible.
    
    dx = abs(sx_tile - tx_tile)
    dy = abs(sy_tile - ty_tile)
    
    # The distance is indeed max(dx, dy).
    # Let's verify with Sample 1: S(5,0), T(2,5)
    # S: x=5, y=0 -> y even, tx = 5//2 = 2. Tile (2, 0)
    # T: x=2, y=5 -> y odd, tx = (2+1)//2 = 1. Tile (1, 5)
    # dx = |2-1| = 1, dy = |0-5| = 5. max(1, 5) = 5. Correct.
    
    # Sample 2: S(3,1), T(4,1)
    # S: x=3, y=1 -> y odd, tx = (3+1)//2 = 2. Tile (2, 1)
    # T: x=4, y=1 -> y odd, tx = (4+1)//2 = 2. Tile (2, 1)
    # dx = 0, dy = 0. max(0, 0) = 0. Correct.
    
    print(max(dx, dy))

if __name__ == '__main__':
    solve()