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
    # This means for a fixed j, the tiles are horizontal pairs starting at even (i+j).
    # Let's transform the coordinates to make the grid uniform.
    # Consider a coordinate system where each "unit" is a 2x2 block.
    # However, a simpler way is to observe the Manhattan distance and the parity.
    
    # Let dx = |sx - tx| and dy = |sy - ty|.
    # Moving vertically (changing y) always crosses a tile boundary unless we are 
    # moving within the same 2x1 tile, but 2x1 tiles are horizontal.
    # So every vertical move of 1 unit costs 1 toll, unless we are already in the target tile.
    # Actually, the structure is:
    # Row j consists of tiles {(0,j), (1,j)}, {(2,j), (3,j)} if j is even.
    # Row j consists of tiles {(-1,j), (0,j)}, {(1,j), (2,j)} if j is odd.
    
    # Let's map (x, y) to the tile index (X, Y).
    # Y = y
    # If y is even, X = x // 2
    # If y is odd, X = (x - 1) // 2 (since the tile is {1, 2}, {3, 4}...)
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # If j is even: i=0, 2, 4... are starts of tiles. X = i // 2.
    # If j is odd: i=-1, 1, 3... are starts of tiles. X = (i+1) // 2.
    
    # Let's use a different approach.
    # To move from (sx, sy) to (tx, ty), the minimum cost is:
    # cost = max(0, (abs(sx - tx) + 1) // 2) + abs(sy - ty)
    # But we can optimize by moving diagonally.
    # The distance is actually related to the L1 distance in a transformed space.
    # Let u = x + y and v = x - y.
    # The tiles are aligned with the x-axis.
    # A move in x costs 0 if we stay in the same 2x1 tile, 1 if we cross.
    # A move in y always costs 1.
    
    # Correct logic for this specific tiling problem:
    # The minimum cost to move between (sx, sy) and (tx, ty) is:
    # ceil( abs(sx - tx) / 2 ) + abs(sy - ty)
    # But we can "save" costs by coordinating x and y moves.
    # Specifically, the cost is max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # No, that's for different tilings.
    
    # Let's re-evaluate:
    # To change y by 1, you must pay 1.
    # To change x by 1, you might pay 0 or 1.
    # If you move from (sx, sy) to (tx, ty), you must pay at least |sy - ty|.
    # While moving vertically, you can shift your x-coordinate to align with the target tx.
    # Each vertical step allows you to move to either of the two x-coordinates 
    # that belong to the same tile in the new row for free.
    
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # In each vertical step, we can effectively move 1 unit horizontally for "free" 
    # (by picking the correct tile).
    # So we can cover min(dx, dy) of the horizontal distance using the vertical steps.
    # The remaining horizontal distance is max(0, dx - dy).
    # This remaining distance costs (max(0, dx - dy) + 1) // 2.
    
    # Total cost = dy + (max(0, dx - dy) + 1) // 2
    # Let's test Sample 1: 5 0, 2 5 -> dx=3, dy=5. Cost = 5 + 0 = 5. Correct.
    # Sample 2: 3 1, 4 1 -> dx=1, dy=0. Cost = 0 + (1+1)//2 = 1. 
    # Wait, Sample 2 says 0. Why?
    # (3, 1) and (4, 1). i=3, j=1. i+j = 4 (even). 
    # Rule: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    
    # The parity of (i+j) matters.
    # Let's use the property: Cost = max(dy, (dx + dy + 1) // 2) if we can shift.
    # Actually, the simplest formula for this specific problem is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's check Sample 2: max(0, (1 + 0 + 1) // 2) = 1. Still 1.
    
    # Let's reconsider:
    # If we are at (x, y), we are in tile ( (x + (y % 2)) // 2, y ).
    # To move to (x', y'), the cost is the Manhattan distance in the (X, Y) space.
    # X = (x + (y % 2)) // 2, Y = y.
    # Cost = abs(X_s - X_t) + abs(Y_s - Y_t).
    
    # Sample 1: S(5,0) -> X=(5+0)//2=2, Y=0. T(2,5) -> X=(2+1)//2=1, Y=5.
    # Cost = abs(2-1) + abs(0-5) = 1 + 5 = 6. Still not 5.
    
    # The sample 1 explanation says:
    # (5.5, 0.5) -> (4.5, 0.5) [Left 1, Cost 0]
    # (4.5, 0.5) -> (4.5, 1.5) [Up 1, Cost 1]
    # (4.5, 1.5) -> (3.5, 1.5) [Left 1, Cost 0]
    # (3.5, 1.5) -> (3.5, 4.5) [Up 3, Cost 3]
    # (3.5, 4.5) -> (2.5, 4.5) [Left 1, Cost 0]
    # (2.5, 4.5) -> (2.5, 5.5) [Up 1, Cost 1]
    # Total = 5.
    
    # In this path, he moves 1 unit left for every 1 unit up.
    # This means he can change his X coordinate by 1 for every Y change.
    # The cost is simply max(abs(sx - tx), abs(sy - ty))? No.
    # Let's see: dx=3, dy=5. max(3, 5) = 5.
    # Sample 2: dx=1, dy=0. max(1, 0) = 1. Still not 0.
    
    # Let's use the (X, Y) coordinates but allow diagonal moves.
    # A move in Y costs 1. A move in X costs 1.
    # But a move in Y also changes the parity of Y, which might change X.
    # If we move from (x, y) to (x, y+1), the tile index X changes from (x + (y%2))//2 to (x + ((y+1)%2))//2.
    # This is a "free" horizontal shift.
    
    # Correct logic:
    # To get from (sx, sy) to (tx, ty), you must pay for every vertical level crossed: abs(sy - ty).
    # Additionally, you might need to move horizontally.
    # Each vertical step allows you to effectively move 1 unit horizontally for free 
    # because the tile boundaries shift.
    # The number of horizontal units you can cover for free is abs(sy - ty).
    # The remaining horizontal distance is max(0, abs(sx - tx) - abs(sy - ty)).
    # This remaining distance is covered by horizontal tiles, costing (rem + 1) // 2.
    # However, we must check the parity of the start and end tiles.
    
    # Let's use the property:
    # The distance is abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Sample 1: 5 + max(0, (3 - 5 + 1) // 2) = 5 + 0 = 5.
    # Sample 2: 0 + max(0, (1 - 0 + 1) // 2) = 1.
    # Wait, Sample 2 is 0. This happens because (3,1) and (4,1) are in the same tile.
    # (3,1): i=3, j=1. i+j=4 (even). Tile is {3,1} and {4,1}.
    # So if we are at (3.5, 1.5), we are in the same tile as (4.5, 1.5).
    
    # Let's use the (X, Y) coordinates: X = (x + (y % 2)) // 2, Y = y.
    # To move from (Xs, Ys) to (Xt, Yt):
    # You can move from (X, Y) to (X, Y+1) with cost 1.
    # You can move from (X, Y) to (X+1, Y) with cost 1.
    # But you can also move from (X, Y) to (X, Y+1) and potentially change your X 
    # relative to the grid without paying extra.
    # Actually, the distance is simply:
    # cost = abs(Ys - Yt) + max(0, abs(Xs - Xt) - abs(Ys - Yt) // 2) 
    # No, that's not it.
    
    # Let's use the most reliable method:
    # The cost is abs(sy - ty) + max(0, abs(Xs - Xt) - (abs(sy - ty) + 1) // 2)
    # Sample 1: 5 + max(0, 1 - 3) = 5.
    # Sample 2: 0 + max(0, 0 - 0) = 0. (Xs = (3+1)//2 = 2, Xt = (4+1)//2 = 2)
    # Sample 3: sx=2552608206527595, sy=5411232866732612, tx=771856005518028, ty=7206210729152763
    # dx = 1780752201009567, dy = 1794977862420151
    # Xs = (2552608206527595 + 0)//2 = 1276304103263797
    # Xt = (771856005518028 + 1)//2 = 385928002759014
    # abs(Xs - Xt) = 890376100504783
    # cost = 1794977862420151 + max(0, 890376100504