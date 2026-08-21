The constraint to avoid explicit loops and recursion forces a functional programming style, emphasizing the use of expressions over statements. This encourages the use of high-level abstractions and prevents the use of mutable state within loops, leading to more concise logic for mathematical problems. For this specific problem, I will use `min` and `max` to handle the coordinate transformations and absolute differences, as the solution relies on a closed-form mathematical formula rather than iterative traversal.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is based on the parity of i + j.
    # When i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 blocks aligned horizontally.
    # The pattern shifts every row.
    # Let's transform the coordinates to a system where we can calculate 
    # the distance in terms of tile crossings.
    # In this grid, moving vertically always crosses a tile boundary.
    # Moving horizontally might not if you stay within the 2x1 block.
    
    # The cost to move between (sx, sy) and (tx, ty) can be modeled as:
    # Each vertical step costs 1.
    # Horizontal movement cost depends on whether you are in a 'gap' 
    # created by the staggered bricks.
    
    # A known formula for this specific tiling problem (Manhattan-like distance 
    # on a brick wall) is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # However, the parity of the starting position relative to the brick 
    # alignment matters.
    
    # Let's normalize the coordinates.
    # A tile covers (i, j) and (i+1, j) if i+j is even.
    # This means for a fixed j, the tiles are {(0,1), (2,3), ...} if j is even
    # and {(1,2), (3,4), ...} if j is odd.
    
    # The distance is effectively the number of tiles entered.
    # The optimal strategy is to move diagonally as much as possible.
    # Each diagonal step (1, 1) costs 1 toll.
    # After reaching the same row or column, remaining steps are handled.
    
    # Let dx = abs(sx - tx) and dy = abs(sy - ty)
    # The cost is dy + max(0, (dx - dy + 1) // 2) if we are aligned 
    # with the bricks, but we must account for the starting offset.
    
    # Correct logic for this specific grid:
    # The cost is dy + max(0, (dx - dy + 1) // 2) if we start at a 
    # boundary that allows 'free' horizontal moves.
    # More generally, the cost is:
    # Let's use the property that we can move 2 units horizontally 
    # for the price of 1 vertical move.
    
    # The minimum toll is:
    # dy + max(0, (dx - dy + 1) // 2) if (sx + sy) parity allows.
    # Actually, the simplest general form for this problem is:
    # cost = dy + max(0, (dx - dy + 1) // 2) 
    # But we must check if the start and end points are in the same tile.
    
    # Let's refine:
    # If we move from (sx, sy) to (tx, ty):
    # 1. We must pay at least dy tolls for the vertical distance.
    # 2. In those dy moves, we can cover up to dy + 1 horizontal units 
    #    (by alternating horizontal and vertical moves) without 
    #    extra cost, depending on the parity.
    
    # The precise formula for this grid is:
    # ans = dy + max(0, (dx - dy + 1) // 2)
    # But we must adjust for the fact that if sx+sy is even, 
    # the tile is {sx, sx+1}, and if odd, it's {sx-1, sx}.
    
    # Let's use the coordinate transformation:
    # A point (x, y) is in tile ( (x if (x+y)%2==0 else x-1)//2, y ) 
    # wait, that's not quite right.
    
    # Let's use the logic:
    # The cost is dy + max(0, (dx - dy + 1) // 2) 
    # if we consider the "free" horizontal move available every other step.
    # Specifically, if we move 1 unit up, we can then move 1 unit left or right 
    # for free if that move stays within the 2x1 tile.
    
    # The most robust formula for this problem:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # result = dy + max(0, (dx - dy + 1) // 2)
    # However, we need to check if the start and end are in the same tile.
    # If sx == tx and sy == ty, cost is 0.
    # If they are in the same 2x1 tile, cost is 0.
    
    # Check if in same tile:
    # Tile ID for (x, y):
    # If (x + y) % 2 == 0: tile_x = x // 2 if (x % 2 == 0) else (x-1)//2
    # This is getting complex. Let's use the parity logic.
    # A point (x, y) is in tile T(x, y):
    # If (x + y) % 2 == 0, it's the left half of a tile if x is even, 
    # right half if x is odd.
    # Actually: square (i, j) and (i+1, j) are one tile if i+j is even.
    # For a point (x, y), the square is A_{x,y}.
    # It belongs to tile {A_{x,y}, A_{x+1,y}} if x+y is even.
    # It belongs to tile {A_{x-1,y}, A_{x,y}} if (x-1)+y is even.
    
    # Let's define tile coordinates (tx, ty):
    # if (sx + sy) % 2 == 0:
    #    tile_sx = sx // 2
    # else:
    #    tile_sx = (sx - 1) // 2
    # This is still confusing. Let's use:
    # The tile containing (x, y) is identified by:
    # y_coord = y
    # x_coord = (x if (x+y)%2 == 0 else x-1) // 2
    
    # Let's apply this:
    # s_tile_x = (sx if (sx + sy) % 2 == 0 else sx - 1) // 2
    # s_tile_y = sy
    # t_tile_x = (tx if (tx + ty) % 2 == 0 else tx - 1) // 2
    # t_tile_y = ty
    
    # Now we are moving on a grid of tiles.
    # Moving from (sx, sy) to (tx, ty) in the original grid.
    # The distance is:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # The answer is dy + max(0, (dx - dy + 1) // 2)
    # But we must handle the case where they start in the same tile.
    
    # Wait, the Sample 2: (3, 1) to (4, 1).
    # sx=3, sy=1. sx+sy = 4 (even). A_{3,1} and A_{4,1} are one tile.
    # So (3.5, 1.5) and (4.5, 1.5) are in the same tile. Cost 0.
    # Formula: dx=1, dy=0. 0 + max(0, (1-0+1)//2) = 1. Incorrect.
    
    # The correct logic:
    # If they are in the same tile, 0.
    # Otherwise, the cost is the number of tiles entered.
    # Let's use the tile coordinates:
    # s_tile_x = (sx if (sx + sy) % 2 == 0 else sx - 1) // 2
    # s_tile_y = sy
    # t_tile_x = (tx if (tx + ty) % 2 == 0 else tx - 1) // 2
    # t_tile_y = ty
    # The distance between these tiles in the transformed grid is:
    # dist = abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y)
    # But we can move diagonally in the original grid.
    # A move (1, 1) in the original grid is a move from (tile_x, tile_y) 
    # to (tile_x', tile_y') which might be a diagonal move in the tile grid.
    
    # Let's re-evaluate:
    # In the tile grid, a move of 1 unit vertically costs 1.
    # A move of 1 unit horizontally costs 1.
    # But a diagonal move (1, 1) in the original grid:
    # (x, y) -> (x+1, y+1). 
    # Tile(x, y) -> Tile(x+1, y+1).
    # If x+y is even, Tile(x, y) is {x, x+1}. Tile(x+1, y+1) is {x+1, x+2}.
    # This is a move of 1 unit in tile_x and 1 unit in tile_y.
    # The cost is 1 (for the new tile).
    # So we can move 1 unit in tile_x and 1 unit in tile_y for a cost of 1.
    # This is the Chebyshev distance if we could move diagonally, 
    # but we can only move diagonally in one "direction" of x.
    # Actually, it's simply:
    # cost = max(abs(s_tile_y - t_tile_y), (abs(s_tile_x - t_tile_x) * 2 + abs(s_tile_y - t_tile_y)) // 2)
    # No, the simplest form is:
    # cost = abs(s_tile_y - t_tile_y) + max(0, abs(s_tile_x - t_tile_x) - abs(s_tile_y - t_tile_y))
    # Wait, if we move diagonally, we cover 1 tile_y and 1 tile_x for cost 1.
    # So cost = max(abs(s_tile_y - t_tile_y), abs(s_tile_x - t_tile_x))
    # Let's check Sample 1: (5, 0) to (2, 5)
    # s_tile_x: (5+0)%2=1 -> (5-1)//2 = 2. s_tile_y = 0.
    # t_tile_x: (2+5)%2=1 -> (2-1)//2 = 0. t_tile_y = 5.
    # max(abs(0-5), abs(2-0)) = max(5, 2) = 5. Correct.
    # Sample 2: (3, 1) to (4, 1)
    # s_tile_x: (3+1)%2=0 -> 3//2 = 1. s_tile_y = 1.
    # t_tile_x: (4+1)%2=1 -> (4-1)//2 = 1. t_tile_y = 1.
    # max(abs(1-1), abs(1-1)) = 0. Correct.
    
    # Final formula:
    # s_tx = (sx if (sx + sy) % 2 == 0 else sx - 1) // 2
    # s_ty = sy
    # t_tx = (tx if (tx + ty) % 2 == 0 else tx - 1) // 2
    # t_ty = ty
    # ans = max(abs(s_ty - t_ty), abs(s_tx - t_tx))
    
    # Wait, let's