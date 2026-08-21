The constraint to avoid explicit loops and recursion forces a functional programming style, which encourages the use of declarative transformations over imperative state changes. For this problem, I will use basic arithmetic operations to calculate the Manhattan distance and coordinate transformations, as the "toll" is essentially a measure of how many tile boundaries are crossed. Specifically, I will use `abs()` for distance and `max()`/`min()` to handle the logic of tile transitions without using `if` statements.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is:
    # If i+j is even, (i, j) and (i+1, j) are one tile.
    # This means tiles are 2x1 rectangles.
    # Let's transform the coordinates to a grid where each cell is a tile.
    # A tile can be identified by (I, J) where:
    # J = j
    # I = (i + j) // 2
    # The toll to move between tiles is the L1 distance in this transformed space.
    
    # Current tile coordinates
    # For a point (x+0.5, y+0.5), i=x and j=y.
    # If x+y is even, the tile is {(x,y), (x+1,y)}. 
    # We can represent this tile by the index I = (x+y)//2, J = y.
    # If x+y is odd, the tile is {(x-1,y), (x,y)}.
    # We can represent this tile by the index I = (x+y-1)//2, J = y.
    # In both cases, I = (x + y) // 2 and J = y.
    
    # Wait, let's refine:
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's define tile coordinates (U, V):
    # V = j
    # U = (i + j) // 2
    # If i+j is even, A_{i,j} is at ( (i+j)//2, j )
    # If i+j is odd, A_{i,j} is at ( (i+j-1)//2, j )
    # This is consistent: U = (i + j) // 2.
    
    # The distance between (Us, Vs) and (Ut, Vt) is |Us - Ut| + |Vs - Vt|.
    # However, moving vertically (changing j) might change the U coordinate.
    # Let's check the cost:
    # Moving from (i, j) to (i, j+1):
    # Tile S: ((i+j)//2, j), Tile T: ((i+j+1)//2, j+1)
    # Cost is 1.
    # Moving from (i, j) to (i+1, j):
    # If i+j is even, they are the same tile. Cost 0.
    # If i+j is odd, they are different tiles. Cost 1.
    
    # This is equivalent to the Manhattan distance in the (U, V) space
    # where U = (x + y) // 2 and V = y.
    # Let's verify Sample 1: (5, 0) to (2, 5)
    # S: U = (5+0)//2 = 2, V = 0
    # T: U = (2+5)//2 = 3, V = 5
    # Dist = |2-3| + |0-5| = 1 + 5 = 6. 
    # Sample 1 output is 5. Let's re-evaluate.
    
    # Correct Logic:
    # The cost to move from (sx, sy) to (tx, ty) is:
    # Let dx = tx - sx, dy = ty - sy.
    # The cost is |dy| + (|dx| - (1 if (sx+sy)%2 == 0 and tx > sx else 0) ...)
    # Actually, the simplest way to view this is:
    # Each step in Y always costs 1.
    # Each step in X costs 1 only if we cross a boundary.
    # Boundaries in X are at x = i where i+j is odd.
    # For a fixed j, the boundaries are at ..., -2, 0, 2, ... if j is odd
    # and ..., -1, 1, 3, ... if j is even.
    
    # Let's use the property: 
    # Cost = |sy - ty| + | (sx + sy)//2 - (tx + ty)//2 |
    # Sample 1: |0 - 5| + |(5+0)//2 - (2+5)//2| = 5 + |2 - 3| = 6. Still 6.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty), we can move to (sx, ty) then to (tx, ty).
    # Cost = |sy - ty| + cost_to_move_X(sx, tx, ty).
    # cost_to_move_X is the number of boundaries x=i where i+ty is odd.
    # This is equivalent to the number of integers i between sx and tx such that i+ty is odd.
    
    # Let's use the formula: 
    # The number of odd integers in [min(sx, tx)+1, max(sx, tx)] is:
    # (max(sx, tx) + 1)//2 - (min(sx, tx) + 1)//2 if we shift by ty.
    
    # Let't use a simpler approach:
    # The cost is |sy - ty| + abs((sx + sy)//2 - (tx + ty)//2) 
    # is almost correct, but we must account for the parity of the start/end.
    # The actual distance is |sy - ty| + | (sx + (sy%2))//2 - (tx + (ty%2))//2 |
    # Sample 1: |0-5| + |(5+0)//2 - (2+1)//2| = 5 + |2-1| = 6.
    
    # Correct logic for this specific tiling:
    # The cost is |sy - ty| + | (sx + sy)//2 - (tx + ty)//2 | 
    # BUT, if we move in Y first, we can change the parity of j to make the X move cheaper.
    # The cost is min(|sy - ty| + |(sx + sy)//2 - (tx + ty)//2|, 
    #                |sy - ty| + |(sx + sy + 1)//2 - (tx + ty + 1)//2|)
    # Wait, the Sample 1: (5,0) -> (2,5). 
    # sy-ty = 5. 
    # (sx+sy)//2 = 2. (tx+ty)//2 = 3. Diff = 1. Total = 6.
    # (sx+sy+1)//2 = 3. (tx+ty+1)//2 = 4. Diff = 1. Total = 6.
    # Sample 1 says 5. How?
    # (5,0) is in tile ((5+0)//2, 0) = (2, 0).
    # (2,5) is in tile ((2+5)//2, 5) = (3, 5).
    # We can move (5,0) -> (4,0) [same tile, cost 0] -> (4,5) [cost 5] -> (2,5) [same tile, cost 0].
    # Total cost = 5.
    
    # The rule is: (i, j) and (i+1, j) are same tile if i+j is even.
    # This means for a fixed j, the tiles are {(0,j),(1,j)}, {(2,j),(3,j)} if j is even.
    # And {( -1,j),(0,j)}, {(1,j),(2,j)} if j is odd.
    # This is exactly: i and i+1 are same tile if (i+j) % 2 == 0.
    # This means the tile index is (i + (j % 2)) // 2.
    
    # Let f(x, y) = (x + (y % 2)) // 2
    # Cost = |sy - ty| + |f(sx, sy) - f(tx, ty)|
    # Sample 1: |0 - 5| + |(5 + 0)//2 - (2 + 1)//2| = 5 + |2 - 1| = 6.
    # Wait, if we move to (4,0) first: f(4,0) = 2. Then move to (4,5): f(4,5) = (4+1)//2 = 2.
    # Then move to (2,5): f(2,5) = (2+1)//2 = 1.
    # Total = 0 (X) + 5 (Y) + 0 (X) = 5.
    
    # The cost is |sy - ty| + min(|f(sx, sy) - f(tx, ty)|, 
    #                            |f(sx, ty) - f(tx, ty)|)
    # But we can move X at any j.
    # The cost to move from sx to tx at height j is |f(sx, j) - f(tx, j)|.
    # We want to minimize |sy - ty| + |f(sx, j) - f(tx, j)| for some j between sy and ty.
    # Since f(x, j) only depends on j % 2, we only check j = sy and j = ty.
    
    # Let's check Sample 1:
    # j = 0: |0-5| + |f(5,0) - f(2,0)| = 5 + |2 - 1| = 6
    # j = 5: |0-5| + |f(5,5) - f(2,5)| = 5 + |(5+1)//2 - (2+1)//2| = 5 + |3 - 1| = 7
    # Wait, I'm still getting 6. Let's re-read.
    # Sample 1: (5,0) -> (2,5). 
    # Move left 1: (4,0). 4+0 is even, so (4,0) and (5,0) are same tile. Cost 0.
    # Move up 1: (4,1). Cost 1.
    # Move left 1: (3,1). 3+1 is even, so (3,1) and (4,1) are same tile. Cost 0.
    # Move up 3: (3,4). Cost 3.
    # Move left 1: (2,4). 2+4 is even, so (2,4) and (3,4) are same tile. Cost 0.
    # Move up 1: (2,5). Cost 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # In this path, he changes X whenever (x+y) is even.
    # This means he can move X for free if he is at a y such that x+y is even.
    # This is like the Manhattan distance on a grid where some edges are 0.
    # The distance is |sx - tx| + |sy - ty|, but we can subtract 1 for every 
    # time we move X across an "even" boundary.
    # Actually, the cost is simply:
    # abs( (sx+sy)//2 - (tx+ty)//2 ) + abs(sy - ty)
    # Let',s re-calculate Sample 1: |(5+0)//2 - (2+5)//2| + |0-5| = |2 - 3| + 5 = 6.
    # Still 6. Let me re-read the rule: "When i+j is even, A_{i,j} and A_{i+1,j} are same tile."
    # Sample 1: S=(5,0), T=(2,5).
    # (5,0): i=5, j=0. i+j=5 (odd). A_{5,0} is a tile.
    # (4,0): i=4, j=0. i+j=4 (even). A_{4,0} and A_{5,0} are SAME TILE.
    # So moving from 5 to 