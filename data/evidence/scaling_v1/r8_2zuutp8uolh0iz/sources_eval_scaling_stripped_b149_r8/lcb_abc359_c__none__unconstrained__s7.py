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
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even: tiles are [0,1]U[1,2], [2,3]U[3,4]... (horizontal tiles start at even i)
    # If j is odd: tiles are [-1,0]U[0,1], [1,2]U[2,3]... (horizontal tiles start at odd i)
    
    # Let's transform the coordinates to a simpler space.
    # In the original problem, moving vertically always crosses a tile boundary 
    # unless you are already in the target tile.
    # Moving horizontally might be free if you stay within the 2x1 tile.
    
    # Let's define a coordinate transformation:
    # Each "row" of tiles is 1 unit high.
    # In row j, the tiles are blocks of 2 units wide.
    # The offset of the blocks depends on (j % 2).
    # We can map (x, y) to a grid where each cell is a tile.
    # The width of a tile is 2, height is 1.
    # For a cell (x, y), the tile index is:
    # row = y
    # col = (x + (y % 2)) // 2
    
    # The distance between (sx, sy) and (tx, ty) in this transformed grid:
    # dx = abs(col_s - col_t)
    # dy = abs(sy - ty)
    
    # The cost to move between two tiles in a grid (where you can move any 
    # distance in one direction) is essentially the L1 distance, 
    # but the "cost" of a move is 1 per tile entered.
    # However, the problem says "Choose a direction and a positive integer n".
    # This means one move can cover many tiles, but you pay for EACH tile entered.
    # Actually, the rule "Each time he enters a tile, he pays a toll of 1" 
    # implies the cost is simply the number of tile boundaries crossed.
    
    # Let's re-evaluate:
    # To get from (sx, sy) to (tx, ty):
    # 1. Vertical distance is always dy = abs(sy - ty).
    # 2. Horizontal distance in terms of tiles:
    #    The tiles are shifted. The cost to move horizontally depends on 
    #    whether the start and end points are in the same tile.
    
    # Let's use the property:
    # Cost = max(abs(sy - ty), abs(sx + (sy%2) - (tx + (ty%2))) // 2) 
    # Wait, the movement is axis-aligned. 
    # The cost is actually:
    # If we move from (sx, sy) to (tx, ty), the minimum cost is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (some overlap))
    
    # Correct logic for this specific tiling:
    # The distance is simply:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # No, that's not quite right. 
    # Let's use the coordinate transformation:
    # A point (x, y) belongs to tile ( (x + (y%2)) // 2, y )
    # Let X1 = (sx + (sy % 2)) // 2, Y1 = sy
    # Let X2 = (tx + (ty % 2)) // 2, Y2 = ty
    # The distance is abs(X1 - X2) + abs(Y1 - Y2)
    # But we must check if the starting tile and ending tile are the same.
    # If (X1, Y1) == (X2, Y2), cost is 0.
    # Otherwise, the cost is abs(X1 - X2) + abs(Y1 - Y2).
    # Wait, if we move diagonally in the tile-grid, we still pay for each tile.
    # The problem says we can move n units in ONE direction.
    # If we move from (X1, Y1) to (X2, Y2), the minimum tiles entered is
    # abs(X1 - X2) + abs(Y1 - Y2).
    
    # Let's test Sample 1: 5 0 -> 2 5
    # X1 = (5 + 0) // 2 = 2, Y1 = 0
    # X2 = (2 + (5%2)) // 2 = (2 + 1) // 2 = 1, Y2 = 5
    # Cost = abs(2 - 1) + abs(0 - 5) = 1 + 5 = 6. 
    # Sample 1 output is 5. Why?
    # Because you can move to a tile that allows a "cheaper" horizontal jump.
    # In the tile grid, moving from (X1, Y1) to (X2, Y2) costs abs(X1-X2) + abs(Y1-Y2).
    # But you can change your X-coordinate "for free" if you are in a 2x1 tile.
    # Actually, the cost is simply:
    # cost = abs(Y1 - Y2) + abs(X1 - X2)
    # But you can reduce this by 1 if you start and end in a way that 
    # the vertical movement covers the horizontal shift.
    
    # Let's reconsider:
    # The cost is simply the L1 distance in the transformed coordinate system:
    # cost = abs(Y1 - Y2) + abs(X1 - X2)
    # For Sample 1: X1=2, Y1=0; X2=1, Y2=5. Cost = 1 + 5 = 6.
    # Still 6. Let's re-read. "Each time he enters a tile, he pays a toll of 1."
    # If he is already in a tile, he doesn't pay for it.
    # So the cost is the number of boundaries crossed.
    # In the transformed grid, the distance is abs(X1-X2) + abs(Y1-Y2).
    # If we start at (X1, Y1) and end at (X2, Y2), the number of boundaries 
    # crossed is abs(X1-X2) + abs(Y1-Y2).
    # For Sample 1: abs(2-1) + abs(0-5) = 6. 
    # Wait, the sample says 5. 
    # (5,0) is in tile ((5+0)//2, 0) = (2, 0).
    # (2,5) is in tile ((2+1)//2, 5) = (1, 5).
    # If he moves from (2,0) to (1,0) [cost 1], then (1,0) to (1,5) [cost 5], total 6.
    # If he moves from (2,0) to (2,5) [cost 5], then (2,5) to (1,5) [cost 1], total 6.
    # How to get 5?
    # If he moves from (2,0) to (2,1), he enters tile ( (2+1)//2, 1 ) = (1, 1).
    # So moving vertically can change the X-coordinate of the tile!
    # Tile at (x, y) is ((x + (y%2)) // 2, y).
    # From (2, 0), moving to y=1 puts him in tile ((5 + 1)//2, 1) = (3, 1).
    # This is the key.
    
    # Let's use the property:
    # The cost is simply abs(X1 - X2) + abs(Y1 - Y2) 
    # BUT, you can move from (X, Y) to (X, Y+1) and the new tile is 
    # ((x + (Y+1)%2)//2, Y+1).
    # This is equivalent to saying the distance is:
    # cost = abs(Y1 - Y2) + abs((sx + (sy%2))//2 - (tx + (ty%2))//2)
    # Actually, the simplest formula for this problem is:
    # cost = abs(sy - ty) + abs((sx + sy)//2 - (tx + ty)//2)
    # Let's check Sample 1: abs(0 - 5) + abs((5+0)//2 - (2+5)//2) = 5 + abs(2 - 3) = 6.
    # Still 6. Let's try: abs(sy - ty) + abs((sx + (sy%2))//2 - (tx + (ty%2))//2)
    # Sample 1: 5 + abs(2 - 1) = 6.
    
    # Let's try another approach:
    # The cost is simply abs(sy - ty) + abs( (sx + sy)//2 - (tx + ty)//2 )
    # Wait, the sample 1 explanation:
    # (5,0) -> (4,0) [Same tile, cost 0]
    # (4,0) -> (4,1) [New tile, cost 1]
    # (4,1) -> (3,1) [Same tile, cost 0]
    # (3,1) -> (3,4) [New tiles, cost 3]
    # (3,4) -> (2,4) [Same tile, cost 0]
    # (2,4) -> (2,5) [New tile, cost 1]
    # Total = 1 + 3 + 1 = 5.
    
    # Let's trace the tiles:
    # (5,0): i=5, j=0. i+j=5 (odd). Tile is A_{5,0} U A_{6,0} NO.
    # Rule: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # (5,0): i=5, j=0. i+j=5 (odd). A_{5,0} is its own tile? No, A_{4,0} and A_{5,0} are one tile because 4+0=4 (even).
    # So (5,0) is in tile {A_{4,0}, A_{5,0}}.
    # (2,5): i=2, j=5. i+j=7 (odd). A_{2,5} and A_{3,5} are one tile because 2+5=7 is odd? 
    # No, the rule is i+j is even. 2+5=7 (odd), so A_{2,5} is NOT paired with A_{3,5}.
    # But A_{1,5} and A_{2,5} are paired because 1+5=6 (even).
    # So (2,5) is in tile {A_{1,5}, A_{2,5}}.
    
    # Let's re-evaluate the tile index:
    # If j is even, tiles are (0,1), (2,3), (4,5)... -> tile index is i // 2
    # If j is odd, tiles are (-1,0), (1,2), (3,4)... -> tile index is (i+1) // 2
    # This is exactly (i + (j%2)) // 2.
    # Sample 1: S(5,0) -> X1 = (5+0)//2 = 2. T(2,5) -> X2 = (2+1)//2 = 1.
    # The cost is abs(Y1-Y2) + abs(X1-X2).
    # But we can move diagonally in the tile-grid!
    # If we move from (X, Y) to (X', Y+1), the cost is 1.
    # This move allows us to change X to X' where X' is the tile index of (x, Y+1).
    # The tile index of (x, Y+1) is (x + ((Y+1)%2)) // 2.
    # If Y is even, index is x//2. If Y+1 is odd, index is (x+1)//