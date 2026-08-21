The constraint to avoid explicit loops and recursion forces a shift from imperative to functional programming styles, encouraging the use of high-level abstractions and mathematical expressions over manual state management. For this problem, which requires calculating a minimum cost based on coordinate distances and parity, a direct mathematical formula is more efficient and concise than any iterative approach.

To solve this, we observe that the tiles are $2 \times 1$ rectangles. Specifically, if $i+j$ is even, the squares $(i, j)$ and $(i+1, j)$ are merged. This creates a brick-like pattern. The cost to move between two points in such a grid depends on the Manhattan distance and the relative parity of the coordinates, as moving "across" the long side of a tile is free, while moving "across" the short side (vertical movement) always incurs a cost.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])
    
    # The cost to move between two points in this specific tiling 
    # can be derived by observing the cost of vertical and horizontal shifts.
    # Vertical movement always costs 1 per unit because every vertical step 
    # enters a new tile.
    # Horizontal movement is free if we are within the same 2x1 tile.
    # The tiles are aligned such that for a fixed y, the boundaries between 
    # tiles are at x = k and x = k+1, where k depends on the parity of y.
    # Specifically, boundaries are at x such that (x + y) is odd.
    
    # Let dx = |sx - tx| and dy = |sy - ty|
    # The minimum cost is dy + (cost to bridge the horizontal gap)
    # The horizontal gap is bridged for 'free' every time we move vertically,
    # as the tile boundaries shift.
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    
    # If we move dy units vertically, we can cover some horizontal distance.
    # In each vertical step, the 'free' horizontal segment shifts.
    # The optimal strategy is to move diagonally in a sense.
    # The cost is max(dy, (dx + dy + 1) // 2) if we consider the grid 
    # as a graph, but more simply:
    # Each vertical step costs 1 and allows us to 'jump' a horizontal boundary.
    # The total cost is dy + max(0, (dx - dy + 1) // 2) if the parity 
    # of the start and end positions relative to the tiles requires an extra step.
    
    # A more robust observation:
    # The cost is dy + max(0, ceil((dx - dy) / 2)) if we are not aligned.
    # Let's refine: 
    # To move from (sx, sy) to (tx, ty):
    # 1. Vertical distance dy always costs dy.
    # 2. After dy vertical steps, we are at (sx, ty). We need to cover dx.
    # 3. In the row ty, tiles are [0,1], [2,3]... if ty is even, or [1,2], [3,4]... if ty is odd.
    # 4. The cost to move horizontally in a row is 0 if both points are in the same tile,
    #    otherwise it's the number of boundaries crossed.
    # However, we can distribute the dy vertical steps to minimize horizontal costs.
    
    # The general formula for this specific tiling problem is:
    # cost = dy + max(0, (dx - dy + 1) // 2) if we account for the parity 
    # of the tiles at the start and end.
    # Actually, the simplest form is:
    # cost = max(dy, (dx + dy + 1) // 2) is for a different grid.
    # For this grid: cost = dy + max(0, (dx - dy + 1) // 2) is almost correct.
    # Let's use the property: cost = max(dy, (dx + dy + 1) // 2) is for 
    # when you can move diagonally. Here you can't.
    # The correct logic: 
    # Each vertical step costs 1. Each 2 horizontal steps cost 1.
    # But a vertical step also 'counts' as a horizontal step in terms of 
    # reaching the destination because it changes the parity of the tile boundary.
    
    # Correct formula for this problem:
    # The cost is dy + max(0, (dx - dy + 1) // 2) if we consider the 
    # relative parity of (sx+sy) and (tx+ty).
    # Let's use the coordinate transformation:
    # The cost is simply (dx + dy) // 2 if we can move diagonally.
    # But we can only move H or V.
    # The minimum cost is actually:
    # ans = dy + max(0, (dx - dy + 1) // 2) if (sx+sy)%2 == (tx+ty)%2 else ...
    # Actually, the most reliable formula for this specific problem is:
    # ans = max(dy, (dx + dy + 1) // 2)
    # Wait, Sample 1: 5 0 to 2 5 -> dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1 -> dx=1, dy=0. max(0, (1+0+1)//2) = 1? No, Sample 2 is 0.
    # Let's re-evaluate. In Sample 2, (3,1) and (4,1) are in the same tile 
    # because i+j = 3+1 = 4 (even), so A_{3,1} and A_{4,1} are one tile.
    
    # Correct logic:
    # A point (x, y) is in tile T(x, y).
    # If (x + y) is even, T(x, y) = T(x + 1, y).
    # This means for a fixed y, the tiles are {(0,y), (1,y)}, {(2,y), (3,y)} ... if y is even.
    # And {(1,y), (2,y)}, {(3,y), (4,y)} ... if y is odd.
    # This is equivalent to saying (x, y) and (x', y) are in the same tile if
    # floor((x + (y % 2)) / 2) == floor((x' + (y % 2)) / 2).
    
    # The distance is the L1 distance in a graph where nodes are tiles.
    # The distance between tile (x, y) and (x', y') is:
    # cost = dy + max(0, (dx - dy + 1) // 2) is still not quite right.
    # Let's use: cost = max(dy, (dx + dy + 1) // 2) if we can move diagonally.
    # But we can't. However, we can move 1 unit V and 1 unit H to change 
    # the tile parity.
    # The actual minimum cost is:
    # If we move dy vertically, we can cover dy horizontal distance for "free"
    # by picking the direction of the horizontal step carefully.
    # The remaining horizontal distance is dx - dy.
    # These remaining steps cost 1 for every 2 units.
    
    # Let's refine:
    # If dx <= dy: cost is dy.
    # If dx > dy: cost is dy + (dx - dy + 1) // 2.
    # But we must check if the start and end tiles are the same.
    # If sx=3, sy=1, tx=4, ty=1: dx=1, dy=0. 
    # (3+1)%2 == 0, so A_{3,1} and A_{4,1} are the same tile. Cost 0.
    # My formula: 0 + (1-0+1)//2 = 1. Still wrong.
    
    # The condition for being in the same tile:
    # y == ty and (x + y) % 2 == 0 and x' == x + 1
    # Or (x + y) % 2 == 1 and x' == x - 1.
    
    # Let's use the property:
    # The cost is dy + max(0, (dx - dy + 1) // 2) 
    # UNLESS we are in the same tile.
    # But the "same tile" logic is already covered if we use:
    # cost = max(dy, (dx + dy + 1) // 2) if we can move diagonally.
    # Wait, the sample 2 says 0. dx=1, dy=0. 
    # If sx=3, sy=1, tx=4, ty=1, then (sx+sy)=4 (even).
    # Rule: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # Here i=3, j=1, i+j=4. So A_{3,1} and A_{4,1} are the same tile.
    
    # Let's use the coordinate transform:
    # A tile is identified by (y, (x + (y % 2)) // 2)
    # Let X = (x + (y % 2)) // 2, Y = y
    # The distance between (X1, Y1) and (X2, Y2) is:
    # We can move from (X, Y) to (X, Y+1) cost 1
    # We can move from (X, Y) to (X+1, Y) cost 1
    # But wait, moving from (X, Y) to (X, Y+1) might be "free" 
    # if the tile at (X, Y+1) is the same as (X, Y)? No, tiles are 2x1.
    # Vertical movement always enters a new tile.
    # Horizontal movement: from X to X+1 costs 1.
    # However, moving from (X, Y) to (X, Y+1) and then to (X', Y+1)
    # might be cheaper.
    
    # The distance is simply:
    # dist = max(abs(Y1 - Y2), (abs(X1 - X2) + abs(Y1 - Y2) + 1) // 2)
    # No, that's for different grids.
    # Let's use: cost = dy + max(0, (dx - dy + 1) // 2)
    # For Sample 2: dx=1, dy=0. (1-0+1)//2 = 1. Still 1.
    # The only way Sample 2 is 0 is if the distance is 0.
    # X1 = (3 + (1%2)) // 2 = (3+1)//2 = 2
    # X2 = (4 + (1%2)) // 2 = (4+1)//2 = 2
    # Y1 = 1, Y2 = 1.
    # Distance between (2, 1) and (2, 1) is 0.
    
    # So the coordinates of the tile are:
    # Y = y
    # X = (x + (y % 2)) // 2
    # The distance between (X1, Y1) and (X2, Y2) is:
    # In one step, we can change Y by 1 (cost 1) and X can stay same or change by 1.
    # This is because moving from (X, Y) to (X, Y+1) is 1 toll.
    # From (X, Y+1), we are in a tile that covers two x-coordinates.
    # One of those x-coordinates is the same as the one in (X, Y).
    # So we can move (X, Y) -> (X, Y+1) and effectively move horizontally.
    # This means the distance is max(abs(Y1 - Y2), abs(X1 - X2)).
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5
    # X1 = (5 + 0)//2 = 2, Y1 = 0
    # X2 = (2 + (5%2))//2 = (2+1)//2 = 1, Y2 = 5
    # max(|2-1|, |0-5|) = max(1, 5) = 5. Correct.
    # Sample 2: sx=3, sy=1, tx=4, ty=1
    # X1 = (3