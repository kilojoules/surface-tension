```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), ... if j is even
    # and pairs (1,2), (3,4), ... if j is odd.
    
    # Let's define a coordinate transformation to a space where 
    # the cost is simply the L1 distance.
    # In the original grid, moving vertically always crosses a tile boundary 
    # (cost 1), unless you are already in the target tile.
    # Moving horizontally might be free if you stay within the 2x1 tile.
    
    # The cost to move from (sx, sy) to (tx, ty) can be modeled as:
    # Each vertical step costs 1.
    # Each horizontal step costs 1, but we get a "discount" based on the 
    # parity of the row and the alignment of the 2x1 tiles.
    
    # A more robust approach:
    # The distance is max(|sy - ty|, (|sx - tx| + 1) // 2) is NOT correct.
    # The correct logic for this specific tiling (bricks) is:
    # The cost is the sum of vertical distance and a modified horizontal distance.
    # Specifically, the cost is:
    # abs(sy - ty) + (abs(sx - tx) + 1) // 2 if the parity allows a shortcut,
    # but actually, the simplest closed form for this specific problem is:
    # cost = abs(sy - ty) + (abs(sx - tx) + 1) // 2 
    # Wait, that's for a different tiling. Let's re-evaluate.
    
    # For this tiling:
    # If we move from (sx, sy) to (tx, ty):
    # Vertical distance is always paid: abs(sy - ty).
    # Horizontal distance: since tiles are 2x1, we can cover 2 units of x 
    # with 1 unit of y change.
    # The minimum cost is actually:
    # max(abs(sy - ty), (abs(sx - tx) + 1) // 2) is for a different problem.
    # For this one, the answer is:
    # abs(sy - ty) + (abs(sx - tx) + 1) // 2 is also not quite it.
    
    # Correct logic:
    # To move horizontally by dx, you need at least (dx+1)//2 tiles.
    # However, you can "piggyback" horizontal movement on vertical movement.
    # Each vertical move allows you to change your "parity" relative to the bricks.
    # The cost is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The cost is dy + max(0, (dx + 1) // 2 - (some value based on dy))
    # Actually, the simplest derivation for this specific tiling is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2)
    # No, that's not it either.
    
    # Let's use the property:
    # The cost is (abs(sx - tx) + abs(sy - ty) + 1) // 2 
    # if we can optimize. But we can't move diagonally.
    # The actual minimum cost is:
    # abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2) 
    # is still wrong.
    
    # Let's use the coordinate transformation:
    # The cost is simply:
    # abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2) 
    # is for a different tile.
    
    # For THIS tile:
    # The cost is: abs(sy - ty) + (abs(sx - tx) + 1) // 2 
    # BUT you can reduce the horizontal cost by 1 if you move vertically.
    # The correct formula is:
    # ans = abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2)
    # Wait, Sample 1: 5 0 to 2 5. dx=3, dy=5. 
    # 5 + max(0, (3+1)//2 - (5+1)//2) = 5 + max(0, 2 - 3) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0.
    # 0 + max(0, (1+1)//2 - (0+1)//2) = 0 + max(0, 1 - 0) = 1. 
    # But Sample 2 says 0. Why?
    # Because at (3,1), i+j = 3+1 = 4 (even). So A_{3,1} and A_{4,1} are one tile.
    # Thus moving from 3.5, 1.5 to 4.5, 1.5 is free.
    
    # Let's use the parity:
    # A point (x, y) is in tile:
    # If (x+y) is even, it's paired with (x+1, y).
    # This means the tile ID is ( (x + (x+y)%2), y ) if we group by 2s.
    # More simply: the tile is identified by ( (x + (x+y)%2) // 2, y )
    
    # Let X = (sx + (sx + sy) % 2) // 2
    # Let Y = sy
    # Let X2 = (tx + (tx + ty) % 2) // 2
    # Let Y2 = ty
    # The distance is abs(X - X2) + abs(Y - Y2)
    # But we must account for the fact that moving Y changes the parity of X.
    # The correct transformation for this specific brick pattern is:
    # Cost = abs(sy - ty) + abs((sx + (sx + sy) % 2) // 2 - (tx + (tx + ty) % 2) // 2)
    # Let's check Sample 2: sx=3, sy=1, tx=4, ty=1
    # X = (3 + (3+1)%2)//2 = (3+0)//2 = 1
    # X2 = (4 + (4+1)%2)//2 = (4+1)//2 = 2
    # Wait, (4+1)//2 is 2. 2-1 = 1. Still 1.
    
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # For sy=1: i+1 is even when i is odd. So A_{1,1}&A_{2,1}, A_{3,1}&A_{4,1} are tiles.
    # (3,1) and (4,1) are indeed in the same tile.
    # The tile index for A_{i,j} is:
    # If (i+j) is even, tile is ( (i // 2), j ) if j is even, or ( (i-1)//2, j ) if j is odd.
    # Actually: tile_x = (i + (i + j) % 2) // 2
    # Sample 2: i=3, j=1 -> (3 + (4)%2)//2 = 3//2 = 1.
    # i=4, j=1 -> (4 + (5)%2)//2 = 5//2 = 2. Still different.
    
    # Let's use the property: tile_x = (i + (i + j) % 2) // 2 is wrong.
    # If i+j is even, i and i+1 are together.
    # For j=1: i=1,2 are together; i=3,4 are together.
    # For i=3, j=1: i+j=4 (even). So A_{3,1} and A_{4,1} are one tile.
    # The pair is (i, i+1) where i is the largest even integer <= x if j is even,
    # and i is the largest odd integer <= x if j is odd.
    # This is: tile_x = (i + (j % 2)) // 2
    # Sample 2: sx=3, sy=1 -> (3 + 1)//2 = 2.
    # tx=4, ty=1 -> (4 + 1)//2 = 2.
    # Cost = abs(2-2) + abs(1-1) = 0. Correct!
    # Sample 1: sx=5, sy=0 -> (5 + 0)//2 = 2. Y=0.
    # tx=2, ty=5 -> (2 + 1)//2 = 1. Y=5.
    # Cost = abs(2-1) + abs(0-5) = 1 + 5 = 6. 
    # But sample output is 5. Why?
    # Because you can move to a tile that allows a "cheaper" horizontal jump.
    # The cost is actually the L1 distance in the transformed space:
    # Cost = abs(sy - ty) + abs((sx + sy%2)//2 - (tx + ty%2)//2)
    # Wait, if we move vertically, we can change our X coordinate in the transformed space.
    # The distance is:
    # Let x1 = (sx + sy%2)//2, y1 = sy
    # Let x2 = (tx + ty%2)//2, y2 = ty
    # The cost is abs(y1 - y2) + abs(x1 - x2)
    # But we can potentially save 1 if we move vertically.
    # Looking at the problem, it's a shortest path on a graph.
    # The distance is simply: abs(sy - ty) + abs((sx + sy%2)//2 - (tx + ty%2)//2)
    # Let's re-check Sample 1: sx=5, sy=0, tx=2, ty=5
    # x1 = (5 + 0)//2 = 2, y1 = 0
    # x2 = (2 + 1)//2 = 1, y2 = 5
    # Cost = abs(0-5) + abs(2-1) = 5 + 1 = 6. Still 6.
    
    # Final attempt at the logic:
    # The cost is abs(sy - ty) + abs((sx + sy%2)//2 - (tx + ty%2)//2)
    # BUT, you can move to (sx, sy+1) first.
    # x1_alt = (5 + 1)//2 = 3.
    # Cost = 1 + abs(5-1) + abs(3-1) = 1 + 4 + 2 = 7.
    # What if we move to (sx-1, sy)?
    # sx=4, sy=0 -> x = (4+0)//2 = 2.
    # The only way to get 5 is if the horizontal distance is 0.
    # (5 + 0)//2 = 2. (2 + 5%2)//2 = (2+1)//2 = 1.
    # If we move from sy=0 to sy=5, we can pick the best x-coordinate.
    # The x-coordinate is (i + j%2)//2.
    # For sy=0, x=2. For sy=5, x=1.
    # But for sy=1, x=(5+1)//2 = 3. For sy=2, x=(5+0)//2 = 2.
    # For sy=3, x=(5+1)//2 = 3. For sy=4, x=(5+0)//2 = 2.
    # For sy=5, x=(5+1)//2 = 3.
    # If we are at sy=5, the tile_x for tx=2 is (2+1)//2 = 1.
    # The distance is abs(sy-ty) + abs(tile_x