```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule:
    # If i + j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means tiles are 2x1 horizontal blocks starting at (i, j) where i+j is even.
    # A tile can be identified by (floor(i/2) if (i+j)%2==0 else floor((i-1)/2), j)
    # However, a simpler coordinate transformation is:
    # Let u = i + j and v = i - j.
    # But the most direct way to think about this is:
    # Moving vertically always enters a new tile (cost 1).
    # Moving horizontally might stay in the same tile (cost 0) or enter a new one (cost 1).
    # Specifically, if we are at (x, y), the tile is the pair {(x, y), (x+1, y)} if x+y is even,
    # or {(x-1, y), (x, y)} if x+y is odd.
    # This is equivalent to saying the tile is defined by (floor((x + (x+y)%2)/2), y).
    
    # Let's define a normalized coordinate for the tiles:
    # For a cell (x, y), its tile ID is ( (x + (x+y)%2) // 2, y )
    # Let X = (x + (x+y)%2) // 2 and Y = y.
    # The distance between (X1, Y1) and (X2, Y2) in this grid is |X1 - X2| + |Y1 - Y2|.
    # However, the cost of moving is slightly different.
    # Moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1.
    # But wait, the problem says "Each time he enters a tile, he pays a toll of 1."
    # If he is already in a tile, moving within it is free.
    # The distance is simply the Manhattan distance between the tile coordinates.
    
    # Let's refine the tile coordinate:
    # If x+y is even, the tile covers {x, x+1} at height y.
    # If x+y is odd, the tile covers {x-1, x} at height y.
    # In both cases, the tile's "left" x-coordinate is x if (x+y)%2 == 0 else x-1.
    # Let L(x, y) = x if (x+y)%2 == 0 else x-1.
    # The tile ID is (L(x, y) // 2, y) is wrong because L is already the left edge.
    # The tile ID is simply (L(x, y), y) but since tiles are 2 units wide, 
    # we can use L(x, y) // 2.
    
    # Let's use the property:
    # Cost = |Y1 - Y2| + max(0, |X1 - X2| - (1 if starting and ending tiles are the same horizontal tile))
    # Actually, the simplest form is:
    # Let x' = (x + (x+y)%2) // 2
    # Let y' = y
    # The distance is |x'_s - x'_t| + |y'_s - y'_t|
    # But we must account for the fact that if we move vertically, we might land in a tile
    # that covers the x-coordinate we need.
    
    # Correct logic:
    # The cost is |y_s - y_t| + max(0, |(x_s + (x_s+y_s)%2)//2 - (x_t + (x_t+y_t)%2)//2|)
    # Wait, the parity of y affects the x-tile.
    # Let's use the transformation:
    # A cell (x, y) belongs to tile ( (x + (x+y)%2)//2, y )
    # Let X(x, y) = (x + (x+y)%2) // 2
    # The distance is |X(sx, sy) - X(tx, ty)| + |sy - ty|
    # But there is a catch: if we move vertically, we can change our X coordinate 
    # "for free" if the tiles overlap.
    # Actually, the simplest answer is:
    # ans = abs(sy - ty) + max(0, abs(X(sx, sy) - X(tx, ty)) - (1 if sy != ty else 0))
    # No, that's not quite right. 
    # The correct distance is:
    # Let x1 = (sx + (sx + sy) % 2) // 2
    # Let x2 = (tx + (tx + ty) % 2) // 2
    # Result is abs(sy - ty) + max(0, abs(x1 - x2) - (1 if sy != ty else 0))
    # Wait, if sy != ty, we can potentially move to a tile that covers the target x
    # during the vertical transition.
    
    # Let's re-evaluate:
    # The cost is simply abs(X(sx, sy) - X(tx, ty)) + abs(sy - ty)
    # BUT, if sy != ty, we can potentially save 1 unit of cost if the 
    # tiles at the start and end "overlap" in their x-range.
    # Actually, the most robust formula is:
    # cost = abs(sy - ty) + max(0, abs(x1 - x2) - (1 if sy != ty else 0))
    # Let's test Sample 1: 5 0, 2 5
    # sx=5, sy=0 -> (5+1)//2 = 3. X1 = 3, Y1 = 0
    # tx=2, ty=5 -> (2+1)//2 = 1. X2 = 1, Y2 = 5
    # abs(0-5) + max(0, abs(3-1) - 1) = 5 + 1 = 6. (Sample says 5).
    # Something is wrong. Let's re-read.
    # Sample 1: (5,0) to (2,5). 
    # Tile at (5,0): 5+0=5(odd), so tile is {4,5}. X=2.
    # Tile at (2,5): 2+5=7(odd), so tile is {1,2}. X=0.
    # Wait, if x+y is even, A_{i,j} and A_{i+1,j} are one tile.
    # For (5,0): 5+0=5(odd). A_{5,0} is NOT the start of a tile. 
    # A_{4,0} and A_{5,0} are one tile because 4+0=4(even).
    # So (5,0) is in tile {4,0} U {5,0}.
    # For (2,5): 2+5=7(odd). A_{2,5} is NOT the start.
    # A_{1,5} and A_{2,5} are one tile because 1+5=6(even).
    # So (2,5) is in tile {1,5} U {2,5}.
    # Tile coordinates: Start=(2, 0), End=(0, 5). (Using tile_index = x//2 if (x+y)%2==0 else (x-1)//2)
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are {(0,1), (2,3), (4,5)...} if j is even
    # and {(1,2), (3,4), (5,6)...} if j is odd.
    # This is exactly: tile_id = (i + (i+j)%2) // 2
    # For (5,0): (5 + (5+0)%2)//2 = (5+1)//2 = 3.
    # For (2,5): (2 + (2+5)%2)//2 = (2+1)//2 = 1.
    # Distance: |3-1| + |0-5| = 2 + 5 = 7. Still not 5.
    # The sample says 5. Let's trace:
    # (5.5, 0.5) -> (4.5, 0.5) [Left 1]: Same tile {4,0}U{5,0}. Cost 0.
    # (4.5, 0.5) -> (4.5, 1.5) [Up 1]: New tile {4,1}U{3,1} is NOT it. 
    # At y=1, i+j even means i+1 is even, so i is odd.
    # Tiles at y=1 are {1,2}, {3,4}, {5,6}.
    # (4.5, 1.5) is in tile {3,1}U{4,1}. Cost 1.
    # (4.5, 1.5) -> (3.5, 1.5) [Left 1]: Same tile. Cost 0.
    # (3.5, 1.5) -> (3.5, 4.5) [Up 3]: Tiles at y=2,3,4. Cost 3.
    # (3.5, 4.5) -> (2.5, 4.5) [Left 1]: Same tile. Cost 0.
    # (2.5, 4.5) -> (2.5, 5.5) [Up 1]: New tile. Cost 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # Observation: In each step, we can move horizontally for free within a tile.
    # The tiles at y and y+1 are shifted.
    # At y, tiles are [2k, 2k+1] if y even, [2k-1, 2k] if y odd.
    # This means we can change our "tile index" by 1 just by moving vertically.
    # The cost is simply:
    # dist = abs(sy - ty) + max(0, abs(X(sx, sy) - X(tx, ty)) - abs(sy - ty))
    # Wait, that's just max(abs(sy - ty), abs(X(sx, sy) - X(tx, ty)))?
    # Let's check Sample 1: X1=3, X2=1, Y1=0, Y2=5.
    # max(5, abs(3-1)) = 5. Correct!
    # Sample 2: 3 1, 4 1.
    # X1: (3 + (3+1)%0)//2 = 3//2 = 1. (Wait, (3+0)//2 = 1)
    # X2: (4 + (4+1)%2)//2 = (4+1)//2 = 2. (Wait, (4+1)//2 = 2)
    # Wait, let's use the formula: X = (x + (x+y)%2) // 2
    # S(3,1): (3 + (3+1)%2)//2 = (3+0)//2 = 1
    # T(4,1): (4 + (4+1)%2)//2 = (4+1)//2 = 2
    # max(abs(1-1), abs(1-2)) = 1. Sample 2 says 0.
    # Let's re-calculate X for Sample 2:
    # S(3,1): i=3, j=1. i+j=4 (even). So A_{3,1} and A_{4,1} are one tile.
    # T(4,1): i=4, j=1. i+j=5 (odd). But A_{3,1} and A_{4,1} are the same tile.
    # So both are in the same tile. Cost 0.
    # My X formula: X(3,1) = (3 + (3+1)%2)//2 = 3//2 = 1.
    # X(4,1) = (4 + (4+1)%2)//2 = 5//2 = 2.
    # The formula X = (x + (x