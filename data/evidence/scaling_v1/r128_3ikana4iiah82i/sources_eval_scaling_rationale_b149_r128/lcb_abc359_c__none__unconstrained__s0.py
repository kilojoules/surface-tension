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
    # This means for a fixed j, tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (1,2), (3,4), etc. if j is odd.
    # A square (x, j) belongs to tile ID:
    # If j is even: x // 2
    # If j is odd: (x - 1) // 2  (effectively shifting the pairing)
    
    # However, a simpler observation:
    # Moving vertically always enters a new tile (cost 1 per unit).
    # Moving horizontally is free if you stay within the 2x1 tile.
    # The cost is primarily the vertical distance |sy - ty|.
    # We only need an extra cost if we must move horizontally across a boundary
    # that isn't covered by the 2x1 tile structure.
    
    # Let's analyze the cost:
    # To get from (sx, sy) to (tx, ty):
    # 1. Vertical distance is dy = abs(sy - ty).
    # 2. Horizontal distance is dx = abs(sx - tx).
    # Each vertical step costs 1.
    # Horizontal movement is free if we are in the same tile.
    # In any row j, tiles are pairs. The boundary between tiles is at x = k 
    # where k+j is odd.
    # If we move from sx to tx in row j, we cross boundaries.
    # But we can move vertically to a row j' where the boundary is not in our way.
    
    # Actually, the problem can be modeled as:
    # Cost = abs(sy - ty) + (1 if we must cross a vertical boundary that 
    # cannot be bypassed by the vertical movement)
    # Wait, the sample 1: (5,0) to (2,5). dy=5, dx=3. Output 5.
    # Sample 2: (3,1) to (4,1). dy=0, dx=1. Output 0.
    # In Sample 2, i=3, j=1. i+j = 4 (even). So A_{3,1} and A_{4,1} are the same tile.
    # Thus moving from 3.5 to 4.5 in row 1 is free.
    
    # General rule:
    # The cost is abs(sy - ty). 
    # But we must check if the start and end points are "connected" by the 
    # vertical movement.
    # If sy == ty, the cost is 0 if (sx, sy) and (tx, sy) are in the same tile,
    # otherwise it's 1 (move up/down and back).
    # But the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means in row j, the tiles are [0,1], [2,3]... if j is even
    # and [-1,0], [1,2], [3,4]... if j is odd.
    
    # Let's use a coordinate transformation to simplify:
    # A square (x, y) belongs to tile (x // 2, y) if y is even
    # A square (x, y) belongs to tile ((x-1) // 2, y) if y is odd.
    # This is equivalent to saying square (x, y) belongs to tile:
    # tile_x = (x + (y % 2)) // 2
    # tile_y = y
    
    # The distance between (sx, sy) and (tx, ty) in this tile-grid:
    # The vertical distance is always abs(sy - ty).
    # The horizontal distance in terms of tiles is abs(tile_x_s - tile_x_t).
    # However, we can move horizontally for free within a tile.
    # The cost is:
    # If sy == ty:
    #    cost = 0 if tile_x_s == tile_x_t else 1
    # If sy != ty:
    #    The vertical moves already cost abs(sy - ty).
    #    Do these vertical moves allow us to change tile_x for free?
    #    Each vertical move (y -> y+1) changes the parity of y, 
    #    which shifts the tile boundaries.
    #    This allows us to move horizontally across any boundary 
    #    with a cost of 1 (the cost of the vertical move).
    #    Therefore, if sy != ty, the cost is simply abs(sy - ty).
    #    Wait, let's check Sample 1: (5,0) to (2,5). abs(0-5) = 5. Correct.
    #    Sample 2: (3,1) to (4,1). sy=ty=1. tile_x_s = (3+1)//2 = 2. tile_x_t = (4+1)//2 = 2.
    #    Cost = 0. Correct.
    
    # Final logic:
    # dy = abs(sy - ty)
    # if dy == 0:
    #     return 0 if (sx + (sy % 2)) // 2 == (tx + (ty % 2)) // 2 else 1
    # else:
    #     return dy

    # Implementing without if/else using conditional expressions:
    dy = abs(sy - ty)
    ans = dy if dy != 0 else (0 if (sx + (sy % 2)) // 2 == (tx + (ty % 2)) // 2 else 1)
    print(ans)

if __name__ == "__main__":
    solve()