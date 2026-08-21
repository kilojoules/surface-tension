```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is based on (i + j) % 2.
    # If i + j is even, (i, j) and (i+1, j) are one tile.
    # This means for a fixed j, the tiles are horizontal 2x1 blocks.
    # Specifically, for row j, the boundaries between tiles are at x = k where k % 2 == (j % 2).
    # To move from (sx, sy) to (tx, ty), we want to minimize the number of tiles entered.
    
    # Let's transform the coordinates to a system where the grid is uniform.
    # Consider the "dual" grid of tile boundaries.
    # A move in x costs 0 if we stay within the same 2x1 tile.
    # A move in y always costs 1 per unit distance unless we are already in the target tile.
    
    # Let f(x, y) be the index of the tile in row y.
    # In row y, the tile boundaries are at x = k such that k % 2 == y % 2.
    # The tile index can be represented as g(x, y) = (x - (y % 2)) // 2.
    
    # However, a simpler way to think about this is:
    # To move from (sx, sy) to (tx, ty), the total cost is:
    # cost = sum_{j=min(sy, ty)}^{max(sy, ty)-1} (1 if we cross a boundary in row j else 0)
    # This is tricky because we can move horizontally first.
    
    # Let's use the property:
    # To get from (sx, sy) to (tx, ty), the minimum cost is:
    # ceil( (abs(sx - tx) + abs(sy - ty)) / 2 )
    # But we must account for the specific alignment.
    
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # If we move only vertically, cost is dy.
    # If we move only horizontally, cost is ceil(dx / 2).
    # The optimal strategy is to zig-zag or move in a way that leverages the 2x1 tiles.
    
    # Correct logic for this specific tiling:
    # The distance is (abs(sx - tx) + abs(sy - ty)) // 2, but we need to check
    # if the start and end points are in the same tile or relative positions.
    
    # Let's normalize: shift coordinates so sx, sy = 0, 0.
    # New tx = tx - sx, ty = ty - sy.
    # The boundary condition depends on the parity of the original sx, sy.
    
    # Let x1, y1 be start and x2, y2 be end.
    # The cost is (abs(x1 - x2) + abs(y1 - y2)) // 2.
    # However, if (x1 + y1) % 2 != (x2 + y2) % 2, we might need an extra 1.
    # Wait, the actual formula for this specific problem is:
    # cost = (abs(sx - tx) + abs(sy - ty) + 1) // 2
    # But we must check if they are in the same tile.
    
    # Let's re-evaluate:
    # If we are at (x, y), moving to (x, y+1) costs 1.
    # Moving to (x+1, y) costs 1 if x % 2 == y % 2, else 0.
    
    # This is equivalent to a graph problem on a grid.
    # The distance between (sx, sy) and (tx, ty) is:
    # dist = (abs(sx - tx) + abs(sy - ty)) // 2
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 0:
    #    If sx // 2 == tx // 2 (and sy == ty), cost is 0.
    # The general formula for this specific tiling is:
    # cost = (abs(sx - tx) + abs(sy - ty)) // 2
    # Then we check the parity of the boundaries.
    
    # Let's use the property:
    # Min cost = (abs(sx - tx) + abs(sy - ty)) // 2
    # If (sx + sy) % 2 == 0: start is at the left of a tile.
    # If (sx + sy) % 2 == 1: start is at the right of a tile.
    # If (tx + ty) % 2 == 0: end is at the left of a tile.
    # If (tx + ty) % 2 == 1: end is at the right of a tile.
    
    # Let A = (sx + sy) % 2 and B = (tx + ty) % 2.
    # If A == 0 and B == 1, we are moving from left-half to right-half.
    # If A == 1 and B == 0, we are moving from right-half to left-half.
    # In both these cases, the Manhattan distance is odd, and the cost is (dist + 1) // 2.
    # If A == B, the Manhattan distance is even, and the cost is dist // 2.
    # UNLESS they are in the same tile, but the formula (0+0)//2 = 0 handles that.
    
    # Actually, the simplest correct formula for this problem is:
    # result = (abs(sx - tx) + abs(sy - ty) + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0)) // 2
    # Wait, if (sx+sy)%2 != (tx+ty)%2, then abs(sx-tx) + abs(sy-ty) is already odd.
    # So (odd + 1) // 2 is the same as ceil(odd / 2).
    # If (sx+sy)%2 == (tx+ty)%2, then abs(sx-tx) + abs(sy-ty) is even.
    # So (even + 0) // 2 is the same as even / 2.
    
    # Let's test Sample 1: 5 0, 2 5. 
    # abs(5-2) + abs(0-5) = 3 + 5 = 8.
    # (5+0)%2 = 1, (2+5)%2 = 1. Parity same.
    # 8 // 2 = 4. But sample output is 5.
    
    # Re-thinking:
    # The cost is 0 if we move within a tile.
    # A tile is {(x, y), (x+1, y)} where x+y is even.
    # This means x and y have the same parity.
    # Let's transform coordinates: u = x + y, v = x - y.
    # A tile consists of (x, y) and (x+1, y).
    # In (u, v) terms: (x+y, x-y) and (x+y+1, x+1-y).
    # This is not simplifying.
    
    # Let's use the property: to change the "parity" of the tile you are in, 
    # you must move vertically or cross a boundary.
    # The minimum cost is actually:
    # cost = (abs(sx - tx) + abs(sy - ty)) // 2
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 1: cost += 1 (if we move right/up)
    # This is getting confusing. Let's use the logic:
    # To move from (sx, sy) to (tx, ty), you must cross at least 
    # abs(sy - ty) horizontal boundaries.
    # And you must cross some vertical boundaries.
    
    # Correct logic:
    # The cost is (abs(sx - tx) + abs(sy - ty)) // 2.
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 1, we are moving from 
    # the "left" square of a tile to the "right" square of a tile.
    # If we only move right and up, we enter a new tile every time x+y increases by 1.
    # The number of tiles entered is ceil( (abs(sx-tx) + abs(sy-ty)) / 2 ).
    # But we can move "backwards" to align ourselves.
    
    # Let's use the coordinate transformation:
    # Each tile can be identified by ( (x + (x+y)%2) // 2, y )
    # Let X = (x + (x+y)%2) // 2, Y = y.
    # Moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1.
    # Wait, if we are at (X, Y), we are in the tile covering { (2X, Y), (2X+1, Y) } if Y is even,
    # or { (2X-1, Y), (2X, Y) } if Y is odd.
    
    # Let's use the most reliable formula for this problem:
    # The distance is (abs(sx - tx) + abs(sy - ty)) // 2.
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 1, we need to check if we can 
    # reach the target without an extra step.
    # Actually, the simplest way to think about it:
    # You pay 1 whenever you move into a new tile.
    # You can move for free within a tile.
    # A tile is {(x, y), (x+1, y)} where x+y is even.
    # This means x and y have the same parity.
    # Let's shift the coordinates so that sx=0, sy=0.
    # New tx = tx - sx, ty = ty - sy.
    # The start is at (0, 0), which is the left half of a tile (since 0+0 is even).
    # The target is at (tx, ty).
    # The cost is (abs(tx) + abs(ty) + 1) // 2 if (tx + ty) is odd, else (abs(tx) + abs(ty)) // 2.
    # But we must check if the start and end are in the same tile.
    # If sx=5, sy=0, tx=2, ty=5:
    # abs(5-2) + abs(0-5) = 3 + 5 = 8. (8+0)//2 = 4.
    # But Sample 1 says 5. Why?
    # (5,0): 5+0=5 (odd) -> Right half of tile {(4,0), (5,0)}
    # (2,5): 2+5=7 (odd) -> Right half of tile {(1,5), (2,5)}
    # To get from right-half to right-half, you might need an extra step.
    
    # Let's use the property:
    # Cost = (abs(sx - tx) + abs(sy - ty)) // 2
    # If (sx + sy) % 2 == 1 and (tx + ty) % 2 == 0: cost += 1
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 1: cost += 0 (can move right for free)
    # No, that's not it.
    
    # Let's use:
    # x_start, y_start, x_end, y_end
    # The number of tiles is:
    # (abs(sx - tx) + abs(sy - ty)) // 2
    # If (sx + sy) % 2 == 1 and (tx + ty) % 2 == 1:
    #    if sx > tx and sy > ty: we might need an extra step? No.
    # Let's use the logic from a known solution:
    # The answer is (abs(sx - tx) + abs(sy - ty)) // 2, 
    # but if (sx + sy) % 2 == 1 and (tx + ty) % 2 == 1, 
    # and we are moving "away" from the free direction, it might increase.
    
    # Correct logic:
    # A move