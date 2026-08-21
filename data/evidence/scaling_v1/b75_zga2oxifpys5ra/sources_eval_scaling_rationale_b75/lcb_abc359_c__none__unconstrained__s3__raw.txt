The constraint to avoid explicit loops and recursion forces a shift from imperative state management to a functional paradigm, emphasizing the transformation of data over the mutation of variables. In this specific problem, since the goal is to calculate a mathematical distance based on a coordinate system of tiles, the solution relies on coordinate transformations and absolute differences rather than iterative traversal.

To solve this, I will map the $(x, y)$ coordinates into a new coordinate system $(u, v)$ where the distance between tiles can be measured linearly. Specifically, the tiling pattern depends on the parity of $x+y$. By transforming the coordinates, the cost to move between two points becomes a function of the Manhattan distance in the transformed space, divided by a scaling factor.

```python
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
    # This means for a fixed j:
    # If j is even: (0,j)-(1,j) are tile 0, (2,j)-(3,j) are tile 1...
    # If j is odd: (-1,j)-(0,j) are tile 0, (1,j)-(2,j) are tile 1...
    
    # We can transform (x, y) to a coordinate system (u, v) where 
    # moving 1 unit in u or v corresponds to crossing a tile boundary.
    # Let u = x + (y % 2), v = y.
    # In the original system, a tile covers two x-coordinates if x+y is even.
    # The horizontal index of the tile is (x + (y % 2)) // 2.
    # The vertical index of the tile is y.
    
    # Let's define the tile coordinates:
    # Tile X: ux = (sx + (sy % 2)) // 2
    # Tile Y: uy = sy
    # Target Tile X: vx = (tx + (ty % 2)) // 2
    # Target Tile Y: vy = ty
    
    # The cost to move between tiles is the Manhattan distance in the 
    # transformed grid, but we must account for the fact that 
    # moving vertically might land us in the same tile or a different one
    # depending on the x-coordinate.
    
    # A simpler transformation:
    # Let X = x, Y = y.
    # The cost is max(|X1-X2|, |Y1-Y2|) is not correct here.
    # The correct approach for this specific tiling is:
    # Cost = (|sx-tx| + |sy-ty|) // 2 if we can optimize.
    # Actually, the distance is simply:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # If we move only vertically, cost is dy.
    # If we move only horizontally, cost is dx // 2 (roughly).
    
    # The most reliable transformation for this problem:
    # New coordinates: X' = x + (y % 2), Y' = y
    # Distance = (|X's - X't| + |Y's - Y't|) // 2
    # Wait, the standard solution for this specific problem (ARC 074 / ABC 066)
    # is based on the transformation:
    # x_new = x + (y % 2)
    # y_new = y
    # But the cost is actually:
    # dist = (abs(x_new_s - x_new_t) + abs(y_new_s - y_new_t)) // 2
    # However, we must handle the parity of the start and end tiles.
    
    # Let's use the coordinate transformation:
    # u = x + (y % 2)
    # v = y
    # The distance is (abs(us - ut) + abs(vs - vt)) // 2
    # But we need to check if the starting tile and ending tile are the same.
    
    us = sx + (sy % 2)
    vs = sy
    ut = tx + (ty % 2)
    vt = ty
    
    # The distance in this transformed grid is the sum of 
    # horizontal and vertical steps. Each step in the transformed 
    # grid represents moving across a tile boundary.
    # Because each tile is 2x1, the distance is:
    ans = (abs(us - ut) + abs(vs - vt)) // 2
    
    # There is a special case: if the start and end are in the same tile,
    # the formula might give 0, which is correct.
    # If they are in adjacent tiles, it should give 1.
    # Let's verify with Sample 1: 5 0 -> 2 5
    # us = 5 + 0 = 5, vs = 0
    # ut = 2 + (5%2) = 3, vt = 5
    # ans = (abs(5-3) + abs(0-5)) // 2 = (2 + 5) // 2 = 3. 
    # Sample 1 output is 5. My transformation is wrong.
    
    # Correct transformation for this problem:
    # The cost is simply the Manhattan distance in a coordinate system 
    # where we scale x by 0.5 and shift based on y.
    # Let's use: 
    # x' = x + (y % 2)
    # y' = y
    # The distance is (|x's - x't| + |y's - y't|) / 2 is for a different problem.
    
    # For this problem:
    # If we move from (sx, sy) to (tx, ty):
    # The cost is max(abs(sx-tx), abs(sy-ty)) is also not it.
    # The correct logic:
    # Each step in Y costs 1.
    # Each 2 steps in X costs 1.
    # But you can move diagonally (1X, 1Y) for a cost of 1.
    # This is equivalent to:
    # cost = max(abs(sy - ty), (abs(sx + (sy%2) - (tx + (ty%2))) + 1) // 2)
    # No, that's not it either.
    
    # Let's use the property:
    # The distance is (abs(sx - tx) + abs(sy - ty)) // 2 
    # if we can move diagonally.
    # Actually, the simplest correct formula for this problem is:
    # x1, y1 = sx, sy
    # x2, y2 = tx, ty
    # ans = (abs(x1 - x2) + abs(y1 - y2) + 1) // 2 
    # Wait, let's try Sample 1: (5-2 + 0-5) = 7. (7+1)//2 = 4. Still not 5.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # You must cover dy = abs(sy - ty) vertical distance.
    # Each vertical move costs 1.
    # You must cover dx = abs(sx - tx) horizontal distance.
    # In one vertical move, you can also move 1 unit horizontally for free
    # if you land in the same tile.
    # The number of horizontal units you can cover "for free" is dy.
    # But you can only cover 1 unit of dx for every 1 unit of dy.
    # And you can cover 2 units of dx for every 1 unit of cost (by moving horizontally).
    
    # Correct logic:
    # You need dy = abs(sy - ty) vertical moves.
    # These dy moves can also cover some horizontal distance.
    # Specifically, in each vertical step, you can move to the 
    # horizontal coordinate of the tile you are entering.
    # The horizontal range of a tile at height y is [i, i+2).
    # The cost is dy + max(0, (dx - dy + 1) // 2) 
    # Let's test Sample 1: dx=3, dy=5. Cost = 5 + max(0, (3-5+1)//2) = 5. Correct.
    # Sample 2: 3 1 -> 4 1. dx=1, dy=0. Cost = 0 + max(0, (1-0+1)//2) = 1. 
    # Wait, Sample 2 output is 0. 
    # In Sample 2, (3,1) and (4,1) are in the same tile because i+j = 3+1 = 4 (even).
    # So A_{3,1} and A_{4,1} are one tile.
    
    # Let's refine:
    # Two points (sx, sy) and (tx, ty) are in the same tile if:
    # sy == ty AND (sx + sy) % 2 == 0 AND tx == sx + 1
    # OR (tx + ty) % 2 == 0 AND sx == tx + 1
    
    # Let's use the coordinate transformation:
    # X = sx + (sy % 2)
    # Y = sy
    # X_t = tx + (ty % 2)
    # Y_t = ty
    # The distance is:
    # dx = abs(X - X_t)
    # dy = abs(Y - Y_t)
    # The cost is:
    # If we move dy vertically, we can cover some dx.
    # Each vertical move costs 1 and can cover 1 unit of X.
    # Any remaining dx must be covered by horizontal moves, where 2 units of X cost 1.
    # Cost = dy + max(0, (dx - dy + 1) // 2)
    # Let's test Sample 2: sx=3, sy=1, tx=4, ty=1
    # X = 3 + 1 = 4, Y = 1
    # X_t = 4 + 1 = 5, Y_t = 1
    # dx = 1, dy = 0. Cost = 0 + (1-0+1)//2 = 1. Still 1.
    
    # Let's try X = (sx + (sy % 2)) // 2
    # Sample 2: X = (3+1)//2 = 2, Y = 1; X_t = (4+1)//2 = 2, Y_t = 1.
    # dx = 0, dy = 0. Cost = 0 + 0 = 0. Correct!
    # Sample 1: 5 0 -> 2 5
    # X = (5+0)//2 = 2, Y = 0
    # X_t = (2+1)//2 = 1, Y_t = 5
    # dx = abs(2-1) = 1, dy = abs(0-5) = 5
    # Cost = 5 + max(0, (1-5+1)//2) = 5. Correct!
    
    # Final Formula:
    # X1 = (sx + (sy % 2)) // 2
    # Y1 = sy
    # X2 = (tx + (ty % 2)) // 2
    # Y2 = ty
    # dx = abs(X1 - X2)
    # dy = abs(Y1 - Y2)
    # ans = dy + max(0, (dx - dy + 1) // 2)
    
    # Wait, the (dx-dy+1)//2 part is for when dx > dy.
    # If dx <= dy, the cost is just dy.
    # If dx > dy, we need (dx-dy) more horizontal distance.
    # Since each horizontal move covers 1 unit of X and costs 1,
    # but we can move 2 units of X by spending 1 (by moving to the 
    # other half of the tile and then moving), the cost is:
    # ans = dy + (dx - dy + 1) // 2 if dx > dy else dy
    
    # Let's double check the logic:
    # In the transformed grid, a move in Y costs 1 and changes X by at most 1.
    # A move in X costs 1 and changes X by 1.
    # This is exactly the distance in a grid where you can move 
    # (1, 1), (1, -1), (-1, 1), (-1, -1) or (0, 1), (0, -1) for cost 1.
    # The distance between (X