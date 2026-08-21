```python
import sys

def solve():
    # Read input and parse integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1 rectangles. 
    # A_{i,j} and A_{i+1,j} are in the same tile if i+j is even.
    # This means for a fixed j:
    # If j is even, tiles are {(0,1), (2,3), (4,5)...} -> pairs (2k, 2k+1)
    # If j is odd, tiles are {(-1,0), (1,2), (3,4)...} -> pairs (2k-1, 2k)
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # For a cell (x, y), its tile ID can be represented as (X, Y).
    # Y is simply y.
    # X depends on the parity of y:
    # If y is even, x is in tile floor(x/2).
    # If y is odd, x is in tile floor((x+1)/2).
    
    # However, a simpler way to think about the distance:
    # Moving vertically always enters a new tile (cost 1 per unit).
    # Moving horizontally:
    # If we are in a tile that spans (x, x+1) and (x+1, x+2), 
    # moving between these two costs 0.
    # Moving to any other x costs 1.
    
    # Let's map (x, y) to a normalized coordinate (X, Y) such that
    # the distance is the L1 distance in the transformed space.
    # The parity of the tile shift is based on (x + y) % 2.
    # A tile covers {(x, y), (x+1, y)} if x+y is even.
    # This is equivalent to saying a tile is defined by (floor((x - (y % 2)) / 2), y).
    
    # Let X(x, y) = (x - (y % 2)) // 2
    # Let Y(x, y) = y
    # The distance between (sx, sy) and (tx, ty) in this tiling is:
    # dist = |X(tx, ty) - X(sx, sy)| + |Y(tx, ty) - Y(sx, sy)|
    # But wait, the cost of moving vertically is 1, and horizontally is 1 per tile.
    # The vertical move is always 1 unit per tile.
    # The horizontal move is 1 unit per 2 coordinates.
    
    # Let's refine:
    # For a fixed y, the tiles are blocks of 2.
    # If y is even, blocks are [0,1], [2,3], ...
    # If y is odd, blocks are [-1,0], [1,2], ...
    # The index of the tile at (x, y) is:
    # If y % 2 == 0: index = x // 2
    # If y % 2 == 1: index = (x + 1) // 2
    
    # Let f(x, y) = (x // 2) if y % 2 == 0 else ((x + 1) // 2)
    # The cost to move from (sx, sy) to (tx, ty) is:
    # abs(f(tx, ty) - f(sx, sy)) + abs(ty - sy)
    # Wait, this is only true if we can move diagonally or if the 
    # parity of the tiles allows a shortcut.
    # Actually, the distance is simply the L1 distance in the transformed space:
    # X = (x + (y % 2)) // 2
    # Y = y
    # Distance = |X2 - X1| + |Y2 - Y1|
    # Let's check Sample 1: (5,0) to (2,5)
    # X1 = (5 + 0)//2 = 2, Y1 = 0
    # X2 = (2 + 1)//2 = 1, Y2 = 5
    # Dist = |1 - 2| + |5 - 0| = 1 + 5 = 6. 
    # Sample 1 output is 5. Why?
    # Because we can change Y and X independently.
    # The cost is min(|X(tx, ty) - X(sx, sy)| + |ty - sy|, 
    #                |X(tx, ty+1) - X(sx, sy)| + |ty+1 - sy|, ...)
    # Actually, the parity of Y affects X.
    # The cost is |X2 - X1| + |Y2 - Y1| where X = (x + (y%2))//2.
    # But we can move to (tx, ty) via (tx, ty-1) or (tx, ty+1).
    # The correct formula for this specific tiling is:
    # Cost = abs((tx + (ty % 2)) // 2 - (sx + (sy % 2)) // 2) + abs(ty - sy)
    # Let's re-test Sample 1: sx=5, sy=0, tx=2, ty=5
    # X1 = (5 + 0)//2 = 2
    # X2 = (2 + (5%2))//2 = (2 + 1)//2 = 1
    # Dist = |1 - 2| + |5 - 0| = 6. Still 6.
    
    # Let's reconsider: the cost is the number of boundaries crossed.
    # A boundary is crossed if we move from tile A to tile B.
    # The tiles are T(x, y) = ((x + (y % 2)) // 2, y).
    # The distance between (X1, Y1) and (X2, Y2) in a grid where 
    # you can move to (X+/-1, Y) or (X, Y+/-1) is |X1-X2| + |Y1-Y2|.
    # However, moving from (X, Y) to (X, Y+1) might land you in the same tile?
    # No, tiles are 2x1, so they only span two X-coordinates. They never span Y.
    # So moving Y always costs 1.
    # The only way to get 5 in Sample 1 is if the X-distance is 0.
    # X1 = (5 + 0)//2 = 2.
    # X2 = (2 + 5%2)//2 = 1.
    # Wait, if we move to Y=4 first: X_at_4 = (2 + 0)//2 = 1.
    # Then X distance is |1 - 2| = 1. Total = 1 + 4 = 5.
    # So we want to minimize |X(tx, ty') - X(sx, sy)| + |ty' - ty| + |ty' - sy|
    # where ty' is the Y-coordinate we use to transition X.
    # Actually, the simplest form is:
    # The cost is |X2 - X1| + |Y2 - Y1|, but we can pick the parity of Y 
    # at the start and end to minimize X.
    # But we are fixed at (sx, sy) and (tx, ty).
    # The cost is simply the L1 distance between the tiles:
    # Tile 1: ( (sx + (sy % 2)) // 2, sy )
    # Tile 2: ( (tx + (ty % 2)) // 2, ty )
    # Distance = |X1 - X2| + |Y1 - Y2|
    # For Sample 1: X1 = (5+0)//2 = 2, Y1 = 0; X2 = (2+1)//2 = 1, Y2 = 5.
    # Dist = |2-1| + |0-5| = 6.
    # Wait, the sample says 5. Let's re-read.
    # "Move left by 1. Pay 0." -> (5,0) to (4,0). 
    # (5,0) and (4,0) are in the same tile because 4+0 is even.
    # So (4,0) is the same tile. Then move up...
    # This means we can move to any x' such that (x', sy) is in the same tile as (sx, sy) for free.
    # The tile containing (sx, sy) is the set of x such that (x + sy)%2 == 0 and floor((x+sy)/2) is constant.
    # This is simply the range [2k - sy, 2k+1 - sy].
    # For (5,0), sy=0, so x is in [4, 5]. We can move to x=4 for free.
    # For (2,5), ty=5, so x is in [2k-5, 2k+1-5]. For x=2, 2+5=7 (odd), 
    # so it's in the tile with x=3 (3+5=8 even). Tile is [2, 3].
    # So we can move to x=3 for free.
    # Now we need the distance between tile X=2 (for x=4,5) and tile X=1 (for x=2,3).
    # X = (x + (y%2)) // 2.
    # For (5,0): X = (5 + 0)//2 = 2.
    # For (2,5): X = (2 + 1)//2 = 1.
    # Distance = |2 - 1| + |5 - 0| = 6.
    # Still 6! Let me re-calculate.
    # (5,0) -> X=2, Y=0.
    # (2,5) -> X=1, Y=5.
    # Is there a way to get 5?
    # If we move to (3,0) first? No, (3,0) is a different tile.
    # What if we move to (4,0) [cost 0], then (4,1) [cost 1], then (3,1) [cost 0], 
    # then (3,2) [cost 1], (3,3) [cost 1], (3,4) [cost 1], (3,5) [cost 1].
    # Total = 1 + 1 + 1 + 1 + 1 = 5.
    # In this path, the X-coordinates of the tiles were:
    # (4,0): X=(4+0)//2 = 2
    # (4,1): X=(4+1)//2 = 2
    # (3,1): X=(3+1)//2 = 2
    # (3,2): X=(3+0)//2 = 1
    # (3,3): X=(3+1)//2 = 2 ... no.
    # Let's use the property: Cost = |X1 - X2| + |Y1 - Y2|
    # But we can change X by 1 by changing Y by 1.
    # The distance is actually:
    # Let dx = X1 - X2, dy = Y1 - Y2.
    # The answer is abs(dx) + abs(dy), but we can reduce it if we can 
    # change X without changing Y? No.
    # If we change Y, we might change X.
    # X(x, y) = (x + (y % 2)) // 2.
    # Notice that X(x, y+1) is either X(x, y) or X(x, y) + 1.
    # This means a vertical move can also be a horizontal move in the X-space.
    # The distance is min(|X(sx, sy) - X(tx, ty)| + |sy - ty|,
    #                   |X(sx, sy) - X(tx, ty+1)| + |sy - (ty+1)|, ...)
    # Actually, the distance is simply:
    # abs(X1 - X2) + abs(Y1 - Y2) is the number of steps.
    # But we can move diagonally in the (X, Y) space!
    # A move (x, y) -> (x, y+1) changes Y by 1 and X by ( (x+1)%2 - x%2 ) / 2.
    # This is a move of (+/- 1, 1) or (0, 1) in (X, Y) space.
    # The minimum cost to move between (X1, Y1) and (X2, Y2