The constraint to avoid explicit loops and recursion forces a shift toward a functional programming style, emphasizing declarative transformations over imperative state changes. For this problem, which requires calculating a minimum cost based on coordinate distances and parity, I will use basic arithmetic and the built-in `max` and `abs` functions to determine the cost of traversing the tiled plane.

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
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 horizontal blocks.
    # In row j, if j is even, tiles are [0,1], [2,3]... (i=0, 2, 4...)
    # In row j, if j is odd, tiles are [-1,0], [1,2]... (i=1, 3, 5...)
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's analyze:
    # For a fixed j:
    # If j is even: i=0 (0+even=even) -> A_{0,j} and A_{1,j} are one tile.
    # If j is odd: i=1 (1+odd=even) -> A_{1,j} and A_{2,j} are one tile.
    
    # Let's define a coordinate system for the tiles.
    # Each tile can be identified by (tile_x, tile_y).
    # Since tiles are 2x1, there are 2 tiles for every 2 units of x.
    # In row j, the tile boundaries are at x = k if (k+j) is odd.
    # Let's transform (x, y) to tile coordinates (tx, ty).
    # For a given y, the tile index can be thought of as (x + (y % 2)) // 2.
    
    # Let's check the cost to move from (sx, sy) to (tx, ty).
    # Moving vertically: each step in y always enters a new tile.
    # Cost = abs(sy - ty).
    # Moving horizontally: 
    # In a row y, tiles are blocks of 2. 
    # The boundary between tiles occurs at x where (x + y) is odd.
    # The number of boundaries crossed between sx and tx in row y is:
    # Let f(x, y) = (x + (y % 2)) // 2
    # Horizontal cost = abs(f(tx, ty) - f(sx, sy)) if we stay in the same row.
    
    # However, we can move vertically and horizontally.
    # The optimal strategy is to move to the row/column that minimizes the cost.
    # The cost is essentially the Manhattan distance in the "tile graph".
    # Let's refine the tile coordinates:
    # A point (x, y) belongs to tile ( (x + (y % 2)) // 2, y )
    # But wait, the problem says we pay 1 every time we ENTER a tile.
    # Starting tile is free.
    
    # Let's use the coordinate transformation:
    # X' = (x + (y % 2)) // 2
    # Y' = y
    # The distance is |X'_s - X'_t| + |Y'_s - Y'_t|.
    # But we can move diagonally in the tile grid by changing y and x.
    # Actually, the cost is simply:
    # cost = abs(sy - ty) + max(0, abs(sx - tx) - (1 if (sx+sy)%2 == 0 and (tx+ty)%2 == 0 and ... else 0))
    # Let's re-evaluate.
    
    # Correct logic for this specific tiling:
    # The distance is max(|sy - ty|, (|sx - tx| + 1) // 2) is NOT correct.
    # The cost is:
    # 1. Vertical distance: dy = abs(sy - ty)
    # 2. Horizontal distance: dx = abs(sx - tx)
    # In each row, we can move 2 units of x for the cost of 1 tile entry, 
    # UNLESS we are already in the tile.
    # The cost is dy + (dx + 1) // 2, but we can optimize by picking 
    # whether to move horizontally at sy or ty.
    
    # Let's use the property:
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) + (1 if (sx+sy)%2 != 0 and (tx+ty)%2 != 0 else 0)) // 2)
    # Actually, the simplest form is:
    # Let x1, y1 = sx, sy and x2, y2 = tx, ty
    # The cost is abs(y1 - y2) + max(0, (abs(x1 - x2) + (1 if (x1+y1)%2 != 0 and (x2+y2)%2 != 0 else 0)) // 2)
    # Wait, let's test Sample 1: 5 0 to 2 5.
    # dx = 3, dy = 5.
    # (5+0)%2 = 1, (2+5)%2 = 1.
    # Cost = 5 + (3 + 1) // 2 = 7. Incorrect. Sample 1 output is 5.
    
    # Re-evaluating:
    # In Sample 1, we can move left 1 (cost 0), then up 1 (cost 1), then left 1 (cost 0)...
    # This means we can "zigzag" to utilize the 2-unit tiles.
    # If we move 1 unit vertically, we change the parity of y, 
    # which shifts the tile boundaries by 1.
    # This allows us to move 1 unit horizontally for free every other vertical step.
    # The cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # is also not quite right.
    
    # The correct logic for this tiling:
    # The cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2) if we can move diagonally.
    # But we can only move L, R, U, D.
    # The cost is actually:
    # cost = abs(sy - ty)
    # remaining_dx = abs(sx - tx)
    # We can cover 2 units of dx for every 1 unit of dy by zigzagging.
    # The number of dx covered by dy is 2 * dy.
    # If abs(sx - tx) > 2 * abs(sy - ty), the additional cost is (abs(sx - tx) - 2 * abs(sy - ty) + 1) // 2.
    # But we must check the parity of the start and end tiles.
    
    # Let's use the coordinate transform:
    # A point (x, y) is in tile ( (x + (y%2))//2, y )
    # To move from (x1, y1) to (x2, y2):
    # The cost is abs(y1 - y2) + max(0, (abs(x1 - x2) - abs(y1 - y2) + 1) // 2)
    # Let's test Sample 1: 5 0, 2 5.
    # abs(0 - 5) + max(0, (abs(5 - 2) - 5 + 1) // 2) = 5 + max(0, -1 // 2) = 5. Correct.
    # Sample 2: 3 1, 4 1.
    # abs(1 - 1) + max(0, (abs(3 - 4) - 0 + 1) // 2) = 0 + (1 + 1) // 2 = 1. 
    # Wait, Sample 2 output is 0. 
    # In Sample 2: (3, 1) and (4, 1). i+j = 3+1 = 4 (even).
    # Rule: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # For i=3, j=1: 3+1=4 (even), so A_{3,1} and A_{4,1} are one tile.
    # Thus (3.5, 1.5) and (4.5, 1.5) are in the same tile. Cost 0.
    
    # Correct logic:
    # Two points (x1, y1) and (x2, y2) are in the same tile if:
    # y1 == y2 AND (x1 + y1) % 2 == 0 AND x2 == x1 + 1
    # OR y1 == y2 AND (x1 + y1) % 2 != 0 AND x2 == x1 - 1
    
    # Let's use the tile coordinates:
    # Tile ID: ( (x + (y % 2)) // 2, y )
    # Let X(x, y) = (x + (y % 2)) // 2
    # Let Y(x, y) = y
    # The distance between (X1, Y1) and (X2, Y2) in the tile graph:
    # From (X, Y), you can move to:
    # (X, Y+1) - cost 1
    # (X, Y-1) - cost 1
    # (X+1, Y) - cost 1
    # (X-1, Y) - cost 1
    # BUT, moving from (X, Y) to (X, Y+1) might be "free" if the 
    # tiles are the same? No, tiles are 2x1 horizontal.
    # So moving vertically always changes the tile.
    # Moving horizontally: (X, Y) to (X+1, Y) costs 1.
    # However, you can move (X, Y) -> (X, Y+1) -> (X, Y) 
    # Wait, the "zigzag" is:
    # (X, Y) -> (X, Y+1) costs 1.
    # From (X, Y+1), the x-range is different.
    # The tile (X, Y) covers x in [2X - (Y%2), 2X + 1 - (Y%2)]
    # The tile (X, Y+1) covers x in [2X - ((Y+1)%2), 2X + 1 - ((Y+1)%2)]
    # These two x-ranges always overlap by 1 unit.
    # So you can move from tile (X, Y) to (X, Y+1) with cost 1.
    # And from (X, Y+1) to (X+1, Y+1) with cost 1.
    # But you can also move from (X, Y) to (X+1, Y) with cost 1.
    # The distance is simply the Manhattan distance in the (X, Y) grid.
    # Dist = abs(X1 - X2) + abs(Y1 - Y2)
    # Let's check Sample 1: (5, 0) and (2, 5)
    # X1 = (5 + 0)//2 = 2, Y1 = 0
    # X2 = (2 + 1)//2 = 1, Y2 = 5
    # Dist = abs(2 - 1) + abs(0 - 5) = 1 + 5 = 6. Still not 5.
    
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Sample 1: (5.5, 0.5) to (2.5, 5.5)
    # Start: x=5, y=0. X1 = (5+0)//2 = 2, Y1 = 0.
    # End: x=2, y=5. X2 = (2+1)//2 = 1, Y2 = 5.
    # If he moves:
    # (5.5, 0.5) -> (4.5, 0.5) : Tile (2,0) -> Tile (2,0). Cost 0.
    # (4.5, 0.5) -> (4.5, 1.5) : Tile (2,0) -> Tile (