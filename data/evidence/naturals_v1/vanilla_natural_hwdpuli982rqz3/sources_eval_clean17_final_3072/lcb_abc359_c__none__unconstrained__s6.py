```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)
    
    # The tiling pattern is:
    # If i + j is even, (i, j) and (i+1, j) are one tile.
    # If i + j is odd, (i, j) and (i, j+1) are one tile.
    # This is a periodic structure with period 2 in both x and y.
    # Let's transform the coordinates to a system where the distance 
    # between tiles is more uniform.
    # Consider the transformation: 
    # u = (x + y) // 2
    # v = (x - y + 1) // 2
    # However, a simpler way to think about this is:
    # To move from (sx, sy) to (tx, ty), we can use a coordinate system
    # where the "cost" is the Manhattan distance in a rotated grid.
    # The distance is max(|(sx+sy) - (tx+ty)|, |(sx-sy) - (tx-ty)|) / 2.
    
    # Let's refine this:
    # Let X = sx + sy and Y = sx - sy
    # Let X' = tx + ty and Y' = tx - ty
    # The minimum toll is actually (max(|X - X'|, |Y - Y'|) + 1) // 2
    # But we must account for the specific parity of the starting tile.
    
    # Correct logic for this specific tiling problem:
    # The distance between two cells (sx, sy) and (tx, ty) in this grid
    # is given by:
    # cost = max(abs(sx + sy - (tx + ty)), abs(sx - sy - (tx - ty))) // 2
    # However, we need to check if the start and end cells are in the same tile.
    # Let's use the property:
    # A cell (x, y) belongs to tile:
    # If (x+y) is even, tile is {(x, y), (x+1, y)}
    # If (x+y) is odd, tile is {(x, y), (x, y+1)}
    
    # Let's map each cell (x, y) to a tile ID (U, V):
    # If (x+y) is even: U = (x+y)//2, V = (x-y)//2. The tile is {(x,y), (x+1,y)}.
    # If (x+y) is odd: U = (x+y)//2, V = (x-y+1)//2. The tile is {(x,y), (x,y+1)}.
    # Wait, a simpler mapping:
    # Let u = (x + y) // 2 and v = (x - y) // 2.
    # The distance between tiles is the L_infinity distance in the (u, v) space.
    
    # Let's use the formula derived for this specific problem:
    # The cost to move from (sx, sy) to (tx, ty) is:
    # max(abs(sx + sy - (tx + ty)), abs(sx - sy - (tx - ty))) // 2
    # But we must handle the "half-steps" carefully.
    
    # Let f(x, y) = ( (x+y)//2, (x-y)//2 )
    # The distance is max(|u1-u2|, |v1-v2|)
    # Let's test with Sample 1: (5,0) to (2,5)
    # S: (5+0)//2 = 2, (5-0)//2 = 2
    # T: (2+5)//2 = 3, (2-5)//2 = -2
    # max(|2-3|, |2-(-2)|) = max(1, 4) = 4. 
    # Sample 1 output is 5. The formula is slightly off.
    
    # Correct approach:
    # The tiles are 2x1 or 1x2.
    # Let's transform coordinates:
    # x' = x + y
    # y' = x - y
    # A move of 1 unit in x or y changes x' by 1 and y' by 1.
    # A tile consists of two adjacent squares.
    # If x+y is even, (x,y) and (x+1,y) are one tile. 
    # These have (x', y') = (x+y, x-y) and (x+1+y, x+1-y) = (x'+1, y'+1).
    # If x+y is odd, (x,y) and (x,y+1) are one tile.
    # These have (x', y') = (x+y, x-y) and (x+y+1, x-y-1) = (x'+1, y'-1).
    
    # In the (x', y') plane, a tile is a pair of points.
    # Moving from one tile to another costs 1.
    # The distance is ceil(max(|x1'-x2'|, |y1'-y2'|) / 2).
    
    x1, y1 = sx, sy
    x2, y2 = tx, ty
    
    # Transform to the (x+y, x-y) coordinate system
    # We need to find the distance between the tiles containing (x1, y1) and (x2, y2)
    # Let g(x, y) be the coordinates of the "top-right" or "bottom-right" of the tile.
    # If (x+y) is even, tile is {(x,y), (x+1,y)}. Let's represent it by (x+1, y).
    # If (x+y) is odd, tile is {(x,y), (x,y+1)}. Let's represent it by (x, y+1).
    
    # Actually, the distance is simply:
    # d = max(abs(sx + sy - (tx + ty)), abs(sx - sy - (tx - ty)))
    # The number of tolls is (d + 1) // 2.
    # Let's check Sample 1: max(|5-7|, |5- (-3)|) = max(2, 8) = 8. (8+1)//2 = 4. Still not 5.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty), the minimum number of tiles entered is:
    # We can move to any (x, y) such that max(|(x+y)-(sx+sy)|, |(x-y)-(sx-sy)|) <= 2*k
    # without paying more than k tolls? No.
    
    # Let's use the property:
    # The distance is (abs(sx - tx) + abs(sy - ty) + 1) // 2 if we can align.
    # Actually, the most reliable way to solve this is:
    # The distance is max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Wait, Sample 1: max(3, 5, (3+5+1)//2) = 5. Correct.
    # Sample 2: max(1, 0, (1+0+1)//2) = 1. Incorrect. Sample 2 is 0.
    
    # Let's use the parity logic:
    # Two cells (x1, y1) and (x2, y2) are in the same tile if:
    # (x1+y1 is even AND x2=x1+1, y2=y1) OR (x1+y1 is odd AND x2=x1, y2=y1+1)
    # This means they are the same tile if:
    # (x1+y1)//2 == (x2+y2)//2 AND (x1-y1)//2 == (x2-y2)//2
    # Let u = (x+y)//2, v = (x-y)//2.
    # The distance is max(|u1-u2|, |v1-v2|).
    # Let's check Sample 1: S(5,0) -> u=2, v=2; T(2,5) -> u=3, v=-2.
    # max(|2-3|, |2-(-2)|) = 4. Still not 5.
    
    # Let's use the most accepted formula for this problem:
    # The distance is max(abs(sx-tx), abs(sy-ty), (abs(sx-tx) + abs(sy-ty) + 1) // 2)
    # But we must subtract 1 if they are in the same tile.
    # No, the simplest correct formula is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # result = max((dx + 1) // 2 + (dy + 1) // 2, (dx + dy + 2) // 3) ... no.
    
    # Correct logic:
    # The distance is max(dx, dy, (dx + dy + 1) // 2) - (1 if same_tile else 0)
    # But "same tile" is tricky.
    # Let's use:
    # u = (x + y) // 2, v = (x - y + 1) // 2
    # The distance is max(abs(u1 - u2), abs(v1 - v2))
    # Sample 1: S(5,0) -> u=2, v=3; T(2,5) -> u=3, v=(-3+1)//2 = -1.
    # max(|2-3|, |3-(-1)|) = 4. Still 4.
    
    # Final attempt at the formula:
    # The distance is max(abs(sx-tx), abs(sy-ty), (abs(sx-tx) + abs(sy-ty) + 1) // 2)
    # If we are in the same tile, the cost is 0.
    # If we are in adjacent tiles, the cost is 1.
    # The formula max(dx, dy, (dx+dy+1)//2) is for a different grid.
    
    # Let's use the property:
    # To get from (sx, sy) to (tx, ty), the cost is:
    # dx = abs(sx - tx), dy = abs(sy - ty)
    # cost = max(dx, dy, (dx + dy + 1) // 2)
    # Then we check if (sx, sy) and (tx, ty) are in the same tile.
    # If they are, the cost is 0.
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    
    # Check if they are in the same tile
    # Tile condition: (i+j even and i+1,j) or (i+j odd and i,j+1)
    # This means (x,y) and (x',y') are same tile if:
    # 1. x=x', y=y'
    # 2. x+y is even, x'=x+1, y'=y
    # 3. x+y is odd, x'=x, y'=y+1
    # 4. x'+y' is even, x=x'+1, y=y'
    # 5. x'+y' is odd, x=x', y=y'+1
    
    is_same = (
        (sx == tx and sy == ty) or
        (sx + sy % 2 == 0 and tx == sx + 1 and ty == sy) or
        (sx + sy % 2 != 0 and tx == sx and ty == sy + 1) or
        (tx + ty % 2 == 0 and sx == tx + 1 and sy == ty) or
        (tx + ty % 2 != 0 and sx == tx and sy == ty + 1)
    )
    # The above is slightly wrong due to % precedence.
    
    # Correcting the "same tile" check:
    is_same = (
        (sx == tx and sy == ty) or
        ((sx + sy) % 2 == 0 and tx == sx + 1 and ty == sy) or
        ((sx + sy) % 2 != 0 and tx == sx and ty == sy + 1) or
        ((tx + ty) % 2 == 0 and sx == tx + 1 and sy == ty) or
        ((tx + ty) % 2 != 0 and sx == tx and