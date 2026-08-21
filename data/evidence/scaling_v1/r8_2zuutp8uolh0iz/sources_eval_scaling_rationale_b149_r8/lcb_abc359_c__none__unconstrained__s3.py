import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is:
    # If i+j is even, (i, j) and (i+1, j) are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, the tiles are {(0,j), (1,j)}, {(2,j), (3,j)} if j is even.
    # In row j, the tiles are {(-1,j), (0,j)}, {(1,j), (2,j)} if j is odd.
    
    # Let's transform coordinates to a system where tiles are always {(2k, j), (2k+1, j)}.
    # If j is odd, we can shift the x-coordinate by -1 to align with the even-row pattern.
    # However, a simpler observation:
    # To move from (sx, sy) to (tx, ty):
    # 1. Vertical movement: Each change in y always enters a new tile.
    #    Cost is |sy - ty|.
    # 2. Horizontal movement: 
    #    Within a row j, tiles are blocks of 2.
    #    The boundary between tiles in row j occurs at x = k where k+j is odd.
    #    Let's normalize: in row j, a tile covers x and x+1 if x+j is even.
    #    This is equivalent to saying the tile ID is (x + (j % 2)) // 2.
    
    # Let's define the tile index for a cell (x, y):
    # If y is even, tile is floor(x/2).
    # If y is odd, tile is floor((x+1)/2).
    # This can be written as: tile_x = (x + (y % 2)) // 2
    
    # The cost to move between (sx, sy) and (tx, ty):
    # The vertical distance is dy = abs(sy - ty).
    # The horizontal distance in terms of tile boundaries is the key.
    # Let's use the transformation: 
    # A point (x, y) belongs to tile ( (x + (y % 2)) // 2, y )
    # But wait, the problem says if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even: tiles are [0,1], [2,3], ... (boundaries at odd x)
    # If j is odd: tiles are [-1,0], [1,2], ... (boundaries at even x)
    
    # Let's map (x, y) to a coordinate system where tiles are always 2x1.
    # Let X = x if y % 2 == 0 else x + 1
    # Then the tile index is X // 2.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # cost = abs(sy - ty) + max(0, (abs(tx_tile - sx_tile)) - (some allowance))
    # Actually, the simplest way to view this is:
    # Each step in Y costs 1.
    # Each step in X costs 1 if it crosses a boundary.
    # In a row, boundaries are every 2 units.
    # The distance in "tile units" horizontally is:
    # dx_tiles = abs((tx + (ty % 2)) // 2 - (sx + (sy % 2)) // 2)
    # But we can move diagonally by changing Y and X.
    # The optimal strategy is to move diagonally as much as possible.
    # The cost is max(abs(sy - ty), abs(tx_tile - sx_tile)) 
    # wait, that's for a different grid.
    
    # Correct logic:
    # Let's use the transformation:
    # u = x + y
    # v = x - y
    # This is not a standard grid. Let's use the property:
    # To move from (sx, sy) to (tx, ty), the minimum cost is:
    # abs(sy - ty) + max(0, abs(tx_tile - sx_tile) - abs(sy - ty))
    # where tx_tile = (tx + (ty % 2)) // 2 and sx_tile = (sx + (sy % 2)) // 2
    # This simplifies to max(abs(sy - ty), abs(tx_tile - sx_tile))
    # Let's check Sample 1: 5 0 to 2 5
    # sx=5, sy=0 -> sx_tile = (5 + 0)//2 = 2
    # tx=2, ty=5 -> tx_tile = (2 + 1)//2 = 1
    # cost = max(abs(0-5), abs(1-2)) = max(5, 1) = 5. Correct.
    # Sample 2: 3 1 to 4 1
    # sx=3, sy=1 -> sx_tile = (3 + 1)//2 = 2
    # tx=4, ty=1 -> tx_tile = (4 + 1)//2 = 2
    # cost = max(abs(1-1), abs(2-2)) = 0. Correct.
    
    sx_tile = (sx + (sy % 2)) // 2
    tx_tile = (tx + (ty % 2)) // 2
    
    print(max(abs(sy - ty), abs(tx_tile - sx_tile)))

if __name__ == "__main__":
    solve()