import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, the tiles are {(0,j), (1,j)}, {(2,j), (3,j)} if j is even.
    # In row j, the tiles are {(-1,j), (0,j)}, {(1,j), (2,j)} if j is odd.
    
    # Let's define a coordinate transformation to a grid of tiles.
    # Each tile can be identified by (tile_x, tile_y).
    # For a square (i, j):
    # If j is even: tile_x = i // 2, tile_y = j
    # If j is odd:  tile_x = (i + 1) // 2, tile_y = j
    # However, the cost is based on entering a NEW tile.
    # Moving within a tile is free. 
    # Moving from tile A to tile B costs 1.
    # The distance is the L1 distance in the tile-grid, but since we can 
    # move any n units, we are looking for the minimum number of tiles 
    # entered. This is equivalent to the distance in a graph where 
    # nodes are tiles and edges connect adjacent tiles.
    
    # Let's refine the tile coordinates:
    # A square (i, j) belongs to tile:
    # tile_y = j
    # tile_x = (i + (1 if j % 2 != 0 else 0)) // 2
    
    # The distance between (sx, sy) and (tx, ty) in this specific tiling:
    # The cost to change y by 1 is always 1 (you must enter a new tile).
    # The cost to change x depends on whether you are in the same tile.
    # The optimal strategy is to move diagonally in the tile-grid.
    # The distance is max(|tile_x1 - tile_x2|, |tile_y1 - tile_y2|) 
    # if we could move diagonally, but we can only move U, D, L, R.
    # Wait, the problem says we can move n units in one of 4 directions.
    # This means we can jump across many tiles in one direction, 
    # but we pay for EVERY tile we enter.
    # If we move from (x, y) to (x, y+n), we enter n tiles.
    # If we move from (x, y) to (x+n, y), we enter some number of tiles.
    
    # Let's reconsider:
    # To get from (sx, sy) to (tx, ty):
    # Vertical distance is always |sy - ty|.
    # Horizontal distance: in each row, tiles are 2 units wide.
    # The number of tiles crossed horizontally is roughly |sx - tx| / 2.
    # Because the tiles shift every row, the "cheapest" way to move 
    # horizontally is to move 1 unit vertically, 1 unit horizontally, 
    # 1 unit vertically... effectively moving diagonally.
    
    # The cost is actually:
    # cost = max(|sy - ty|, (|sx + sy + (sx+sy)%2)//2 - (tx + ty + (tx+ty)%2)//2) ... No.
    
    # Correct logic for this specific tiling:
    # The distance is max(|sy - ty|, (|sx + (sy%2) - (tx + (ty%2))| + 1) // 2)
    # Actually, the simplest form for this problem is:
    # ans = max(abs(sy - ty), (abs(sx - tx) + (1 if (sx+sy)%2 == (tx+ty)%2 and sx%2 != tx%2 else 0)) // 2)
    # Let's use the coordinate transformation:
    # Each tile is a node in a graph. 
    # Tile(i, j) is connected to Tile(i+/-1, j) and Tile(i, j+/-1).
    # The distance is |tile_x1 - tile_x2| + |tile_y1 - tile_y2|.
    # BUT, we can move diagonally in the original plane by 
    # changing x and y by 1, which might only cost 1 tile entry.
    # Example: (0,0) -> (1,1). 
    # Tile(0,0) is {(0,0), (1,0)}. Tile(0,1) is {(-1,1), (0,1)}. Tile(1,1) is {(1,1), (2,1)}.
    # From (0.5, 0.5) [Tile 0,0] to (1.5, 1.5) [Tile 1,1]:
    # Move Right 1 to (1.5, 0.5) [Still Tile 0,0], then Up 1 to (1.5, 1.5) [Tile 1,1].
    # Total cost: 1.
    # This means we can move from Tile(i, j) to Tile(i+1, j+1) with cost 1.
    # This is the Chebyshev distance in the tile-grid: max(|dx|, |dy|).
    
    # Tile coordinates:
    # ty1 = sy
    # tx1 = (sx + (sy % 2)) // 2
    # ty2 = ty
    # tx2 = (tx + (ty % 2)) // 2
    
    # Result is max(abs(ty1 - ty2), abs(tx1 - tx2))
    
    # Let's check Sample 1: 5 0, 2 5
    # ty1 = 0, tx1 = (5 + 0) // 2 = 2
    # ty2 = 5, tx2 = (2 + 1) // 2 = 1
    # max(|0-5|, |2-1|) = max(5, 1) = 5. Correct.
    
    # Sample 2: 3 1, 4 1
    # ty1 = 1, tx1 = (3 + 1) // 2 = 2
    # ty2 = 1, tx2 = (4 + 1) // 2 = 2
    # max(0, 0) = 0. Correct.
    
    # Sample 3: 2552608206527595 5411232866732612, 771856005518028 7206210729152763
    # ty1 = 5411232866732612, tx1 = (2552608206527595 + 0) // 2 = 1276304103263797
    # ty2 = 7206210729152763, tx2 = (771856005518028 + 1) // 2 = 385928002759014
    # max(|5411232866732612 - 7206210729152763|, |1276304103263797 - 385928002759014|)
    # max(1794977862420151, 890376100504783) = 1794977862420151. Correct.

    ty1 = sy
    tx1 = (sx + (sy % 2)) // 2
    ty2 = ty
    tx2 = (tx + (ty % 2)) // 2
    
    print(max(abs(ty1 - ty2), abs(tx1 - tx2)))

if __name__ == "__main__":
    solve()