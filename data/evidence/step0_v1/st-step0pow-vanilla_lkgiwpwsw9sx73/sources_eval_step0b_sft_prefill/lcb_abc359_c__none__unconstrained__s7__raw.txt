```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the x-boundaries of tiles are at:
    # j even: ..., -2, 0, 2, 4, ...
    # j odd: ..., -1, 1, 3, 5, ...
    # In general, the vertical boundaries between tiles in row j are x = k where k % 2 != j % 2.
    
    # Let's transform the coordinates to a system where the grid is uniform.
    # Consider the transformation: 
    # u = x + y
    # v = x - y
    # However, a simpler way is to look at the distance.
    # To move from (sx, sy) to (tx, ty), he must cross some boundaries.
    # Vertical boundaries are x = k where k % 2 != j % 2.
    # Horizontal boundaries are y = k for all k.
    
    # Let's analyze the cost to change x and y.
    # Moving in y direction: every unit step crosses a boundary.
    # Moving in x direction: every unit step might cross a boundary.
    
    # Let dx = tx - sx and dy = ty - sy.
    # The minimum cost is actually related to the Manhattan distance in a transformed space.
    # Specifically, the cost is max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # But we must be careful with the parity.
    
    # Let's use the property:
    # Cost = (abs(sx + sy - (tx + ty)) + 1) // 2 + (abs(sx - sy - (tx - ty)) + 1) // 2
    # No, that's for diagonal grids.
    
    # Correct logic:
    # The tiles are 2x1. 
    # In row j, boundaries are at x = k if k % 2 != j % 2.
    # This is equivalent to saying a boundary exists between (i, j) and (i+1, j) if i % 2 == j % 2.
    # A boundary always exists between (i, j) and (i, j+1).
    
    # Let X = sx + sy and Y = sx - sy
    # Let X' = tx + ty and Y' = tx - ty
    # The distance is (abs(X - X') + abs(Y - Y')) // 2
    
    # Let's verify with Sample 1: 5 0 -> 2 5
    # X = 5, Y = 5; X' = 7, Y' = -3
    # abs(5-7) + abs(5 - (-3)) = 2 + 8 = 10. 10 // 2 = 5. Correct.
    
    # Sample 2: 3 1 -> 4 1
    # X = 4, Y = 2; X' = 5, Y' = 3
    # abs(4-5) + abs(2-3) = 1 + 1 = 2. 2 // 2 = 1. 
    # Wait, Sample 2 output is 0.
    # In Sample 2: (3,1) and (4,1). i=3, j=1. i+j = 4 (even).
    # Rule: When i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # A_{3,1} and A_{4,1} are the same tile. Cost 0.
    
    # The actual distance is:
    # Let x1, y1 be start, x2, y2 be end.
    # The cost is the number of boundaries crossed.
    # Horizontal boundaries: abs(y1 - y2)
    # Vertical boundaries: This is tricky because they shift.
    # Notice that if we move 1 unit in X and 1 unit in Y, we can potentially 
    # cross only one boundary (the horizontal one) if we time it right.
    
    # The distance is actually:
    # cost = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # No, that's not it.
    
    # Let's use the coordinate transformation:
    # New coordinates: u = (x + y), v = (x - y)
    # A move of 1 in x: (u+1, v+1)
    # A move of 1 in y: (u+1, v-1)
    # The boundaries are u = constant and v = constant.
    # Specifically, boundaries are at u = even and v = even.
    # The cost is (abs(u1 - u2) + 1) // 2 + (abs(v1 - v2) + 1) // 2 is also not quite right.
    
    # Let's reconsider:
    # To go from (sx, sy) to (tx, ty), the minimum cost is:
    # abs(sy - ty) + (number of vertical boundaries crossed)
    # We can minimize vertical boundaries by moving diagonally.
    # Each 1-unit move in Y allows us to move 1 unit in X for "free" 
    # (by picking the direction that doesn't cross a vertical boundary).
    
    # Let dx = abs(sx - tx), dy = abs(sy - ty).
    # We can cover min(dx, dy) of the x-distance using the y-distance.
    # The remaining x-distance is dx - min(dx, dy).
    # These remaining steps must be paid for.
    # However, the "free" steps depend on the parity.
    
    # Correct formula for this specific tiling problem:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: 5 0, 2 5. dx=3, dy=5. cost = 5 + max(0, (3-5+1)//2) = 5. Correct.
    # Sample 2: 3 1, 4 1. dx=1, dy=0. cost = 0 + max(0, (1-0+1)//2) = 1. Still 1.
    
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    # This means the boundary is at x = i+1 where i+j is odd.
    # Boundary at x = k if k+j is odd.
    # This means for a fixed j, boundaries are at k = ..., -1, 1, 3, 5 ... if j is even,
    # and k = ..., 0, 2, 4, 6 ... if j is odd.
    
    # If we are at (sx, sy) and move to (tx, ty):
    # The number of vertical boundaries between sx and tx is:
    # count(k between sx and tx such that k % 2 != sy % 2)
    
    # Let f(x, y) be the number of vertical boundaries in row y from 0 to x.
    # f(x, y) = x // 2 if x % 2 == y % 2 else (x + 1) // 2
    # This is actually just (x + (y % 2)) // 2.
    
    # But we can change y to minimize this.
    # The cost is min_{y_path} (sum of horizontal boundaries + sum of vertical boundaries).
    # This is a shortest path problem on a graph.
    # The cost to move from (x, y) to (x+1, y) is 1 if x % 2 == y % 2 else 0.
    # The cost to move from (x, y) to (x, y+1) is 1.
    
    # Let's use the property:
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Let's test Sample 2: 3 1, 4 1. dx=1, dy=0. (3+1)%2=0, (4+1)%2=1.
    # cost = 0 + max(0, (1 - 0 + 1) // 2) = 1. Still 1.
    
    # Wait, Sample 2: S=(3,1), T=(4,1). i=3, j=1. i+j = 4 (even).
    # Rule: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    # My logic: i=3, j=1. i+j=4. Boundary is between A_{i,j} and A_{i+1,j} if i+j is odd.
    # Here 3+1=4 is even, so no boundary.
    
    # Let's use the coordinate transform:
    # Let u = x + y, v = x - y.
    # A boundary is crossed if u changes parity or v changes parity? No.
    # Let's use the property:
    # The cost is (abs(sx - tx) + abs(sy - ty)) // 2, but we must handle the parity.
    # Let dx = abs(sx - tx), dy = abs(sy - ty).
    # If (sx + sy) % 2 == (tx + ty) % 2:
    #    cost = max(dx, dy)
    # Else:
    #    cost = max(dx, dy) + 1 (Wait, no)
    
    # Let's re-evaluate:
    # To move from (sx, sy) to (sx+1, sy), cost is 1 if (sx+sy)%2 == 0 else 0.
    # To move from (sx, sy) to (sx, sy+1), cost is 1.
    # This is equivalent to:
    # Cost = abs(sy - ty) + (number of k such that k % 2 == (sy + (k-sx)) % 2)
    # This is getting complex. Let's use the simplest observation:
    # The tiles are 2x1. You can move 2 units in X for the price of 1 unit in Y.
    # The minimum cost is max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2).
    # Sample 1: max(5, (3 + 5 + 1) // 2) = 5.
    # Sample 2: max(0, (1 + 0 + 1) // 2) = 1. Still 1.
    
    # Let's look at Sample 2 again. S=(3,1), T=(4,1).
    # i=3, j=1. i+j=4 (even). A_{3,1} and A_{4,1} are same tile.
    # My formula says 1, but answer is 0.
    # The condition for boundary is i+j is odd.
    # For S=(3,1), i+j=4. For T=(4,1), i+j=5.
    # The boundary is at x=4 if 4+1 is odd. 5 is odd. So boundary at x=4.
    # But the tile is A_{3,1} and A_{4,1}. The boundary is at x=3 or x=5.
    # A_{i,j} is [i, i+1] x [j, j+1].
    # Tile 1: A_{3,1} U A_{4,1} = [3, 5] x [1, 2].
    # Point S is (3.5, 1.5), Point T is (4.5, 1.5).
    # Both are in [3, 5] x [1, 2]. Cost 0.
    
    # General rule:
    # If i+j is even, (i, j) and (i+1, j) are one tile.
    # This means the