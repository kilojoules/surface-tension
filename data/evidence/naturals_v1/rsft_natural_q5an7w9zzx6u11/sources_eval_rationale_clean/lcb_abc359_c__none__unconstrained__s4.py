The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing the transformation of data. For this problem, which requires calculating a minimum cost based on coordinate distances and parity, I will use direct arithmetic expressions and the `abs` and `max` functions to determine the cost of traversing the tiled plane.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern:
    # When i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 blocks.
    # In row j, if j is even: tiles are [0,1], [2,3], ... (i=0, 2, 4...)
    # In row j, if j is odd: tiles are [-1,0], [1,2], ... (i=1, 3, 5...)
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's analyze row j:
    # If j is even: i=0 (0+j even) -> A_{0,j} & A_{1,j} are one tile.
    # If j is odd: i=1 (1+j even) -> A_{1,j} & A_{2,j} are one tile.
    
    # Let's define a coordinate transformation to a grid of tiles.
    # Each tile can be identified by (tile_x, tile_y).
    # Since tiles are 2x1, we can think of the plane as being divided into 
    # strips of height 1. In each strip, tiles are width 2.
    # The offset of the 2-width tiles shifts by 1 every row.
    
    # A simpler observation:
    # To move from (sx, sy) to (tx, ty):
    # The cost is primarily driven by the vertical distance |sy - ty|.
    # Each vertical move of 1 unit enters a new tile.
    # Horizontal movement within a row is free if you stay in the same tile.
    # If you move across a tile boundary, it costs 1, BUT you can move vertically
    # to a row where the tile boundary is not in your way.
    
    # Let's use the property:
    # The cost to move from (sx, sy) to (tx, ty) is:
    # max(|sx - tx|, |sy - ty|) is NOT correct here because tiles are 2x1.
    # The actual cost is based on the Manhattan distance in a transformed space.
    # Let u = x + y and v = x - y.
    # However, the simplest derivation for this specific tiling is:
    # Cost = max(|sx - tx|, |sy - ty|, (|sx - tx| + |sy - ty| + 1) // 2) 
    # is for 1x1 tiles. For 2x1 tiles:
    # The cost is max(|sy - ty|, (|sx - tx| + |sy - ty|) // 2) 
    # if we consider the parity of the tiles.
    
    # Correct logic for this tiling:
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # The cost is max(dy, (dx + dy + 1) // 2) is for a different problem.
    # For this specific 2x1 tiling:
    # The cost is max(dy, (dx + 1) // 2) if we only move horizontally/vertically?
    # No. Let's re-evaluate.
    # In any row, you can move 2 units horizontally for the cost of 1 vertical move.
    # The minimum cost is max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2) 
    # is actually for a grid where you can move diagonally.
    # Here, you can move any n units. 
    # If you move vertically, you pay 1 per unit.
    # If you move horizontally, you pay 1 per 2 units (roughly).
    # The optimal strategy is to move diagonally in terms of tiles.
    # The cost is max(|sy - ty|, (abs(sx - tx) + 1) // 2) is also not quite it.
    
    # Let's use the coordinate transformation:
    # A point (x, y) belongs to tile ( (x + (y % 2)) // 2, y )
    # Let X = (x + (y % 2)) // 2, Y = y
    # The distance between (X1, Y1) and (X2, Y2) in this tile-grid:
    # You can move from (X, Y) to (X, Y+1) cost 1.
    # You can move from (X, Y) to (X+1, Y) cost 1.
    # But wait, moving from (X, Y) to (X, Y+1) might be "free" if the 
    # tile at (X, Y+1) is the same as (X, Y). But tiles are 2x1 (horizontal).
    # So vertical movement always costs 1.
    # Horizontal movement: moving from tile X to X+1 costs 1.
    # However, you can move from (X, Y) to (X, Y+1) and then (X', Y+1)
    # might be the same tile as (X, Y) if the boundaries shift.
    
    # Let's use the formula: cost = max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Wait, the sample 1: (5,0) to (2,5). dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0. max(0, (1+0+1)//2) = 1? 
    # But Sample 2 output is 0. 
    # In Sample 2: sx=3, sy=1. i+j = 3+1=4 (even). A_{3,1} and A_{4,1} are one tile.
    # So (3.5, 1.5) and (4.5, 1.5) are in the same tile. Cost 0.
    
    # The correct logic:
    # Two points (sx, sy) and (tx, ty) are in the same tile if:
    # sy == ty AND (sx + sy) // 2 == (tx + ty) // 2 AND (sx + sy) % 2 == 0
    # Actually: they are in the same tile if sy == ty and 
    # ((sx + sy) % 2 == 0 and tx == sx + 1) or ((tx + ty) % 2 == 0 and sx == tx + 1)
    
    # Let's use the tile coordinates:
    # Tile X = (x + (y % 2)) // 2
    # Tile Y = y
    # Distance = |X1 - X2| + |Y1 - Y2|
    # But you can move from (X, Y) to (X, Y+1) and the tile index X might change.
    # The cost is actually:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The answer is max(dy, (dx + dy + 1) // 2) is for a different grid.
    # For this grid, the answer is:
    # If sy == ty: cost is 0 if they are in the same tile, else 1 if they are adjacent?
    # No, if sy == ty, you can move to sy+1, then move horizontally, then back to sy.
    # That would cost 2. Or just move horizontally and pay for each tile boundary.
    # The cost to move horizontally in row y is (dx + 1) // 2 if the start/end 
    # are not in the same tile.
    
    # Let's use the property:
    # The distance is max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's check Sample 2: dx=1, dy=0. max(0, (1+0+1)//2) = 1. Still 1.
    # The only way Sample 2 is 0 is if the formula is different.
    # In Sample 2, sx=3, sy=1. i+j = 3+1=4 (even). A_{3,1} and A_{4,1} are one tile.
    # So they are in the same tile. Cost 0.
    # If they were in different tiles in the same row, the cost would be 1 
    # because you can move to the boundary and then to the next tile.
    # Wait, the rule is: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    
    # Correct logic:
    # Let X(x, y) = (x + (y % 2)) // 2
    # Let Y(x, y) = y
    # The distance is the Manhattan distance in the (X, Y) space:
    # dist = abs(X(sx, sy) - X(tx, ty)) + abs(Y(sx, sy) - Y(tx, ty))
    # But you can move diagonally in the (X, Y) space? 
    # No, the moves are strictly horizontal and vertical.
    # A move of n units right:
    # If you are at (x, y), you are in tile (X, Y).
    # Moving right to (x+n, y) crosses some boundaries.
    # The number of tiles entered is X(x+n, y) - X(x, y).
    # A move of n units up:
    # You move from (x, y) to (x, y+n).
    # You enter tiles at y+1, y+2, ..., y+n.
    # That is n tiles.
    # However, you can optimize by picking x such that you stay in the same tile.
    # But the tiles are 2x1 (horizontal), so you ALWAYS enter a new tile when moving vertically.
    # The only way to save is to move horizontally within a tile.
    # This is exactly the Manhattan distance in the (X, Y) grid:
    # Cost = abs(X(sx, sy) - X(tx, ty)) + abs(Y(sx, sy) - Y(tx, ty))
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5
    # X(5, 0) = (5 + 0) // 2 = 2
    # Y(5, 0) = 0
    # X(2, 5) = (2 + 1) // 2 = 1
    # Y(2, 5) = 5
    # Dist = abs(2 - 1) + abs(0 - 5) = 1 + 5 = 6. 
    # Sample 1 output is 5. Why?
    # Because you can move to a position where the X coordinate is the same.
    # If you move from (X, Y) to (X, Y+1), the tile index X might change.
    # Let's re-evaluate:
    # Tile at (x, y) is T(x, y) = ((x + (y % 2)) // 2, y)
    # You can move from (X, Y) to (X', Y) with cost |X - X'|
    # You can move from (X, Y) to (X, Y+1) with cost 1.
    # BUT, the tile at (x, y+1) is T(x, y+1) = ((x + ((y+1) % 2)) // 2, y+1)
    # Notice that for a fixed x, T(x, y) and T(x, y+1) might have the same X coordinate!
    # Example: x=1, y=0 -> X = (1+0)//2 = 0.
    # x=1, y=1 -> X = (1+1)//2 = 1.
    # Example: x=2, y=0 -> X = (2+0)//2 = 1.
    # x=2, y=1 -> X = (2+1)//2 = 1.
    # So if you are at x=2, moving from y=0 to y=1 keeps you in X=1.
    # This means