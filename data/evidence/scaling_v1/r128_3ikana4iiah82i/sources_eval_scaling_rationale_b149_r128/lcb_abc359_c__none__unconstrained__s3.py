The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing expressions over statements. For this problem, the optimal path is determined by the Manhattan distance and the specific layout of the $2 \times 1$ tiles. Specifically, the cost is driven by the change in the $y$-coordinate and whether the $x$-coordinates fall into the same tile. I will use `abs()` for distances and a conditional expression to handle the tile logic.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (1,2), (3,4), etc. if j is odd.
    # A square (x, y) belongs to tile ID:
    # If (x + y) is even, it's the left half of a tile.
    # If (x + y) is odd, it's the right half of a tile.
    # Specifically, the tile index for (x, y) can be represented as:
    # Tile_X = (x if (x + y) % 2 == 1 else (x - 1)) // 2
    # However, a simpler observation:
    # Moving horizontally costs 0 if you stay within the same 2x1 tile.
    # Moving vertically always enters a new tile.
    # The cost is primarily the vertical distance |sy - ty|.
    # We must check if the start and end points are in the same tile.
    
    # Let's define the tile coordinate for (x, y):
    # If (x + y) % 2 == 0, the tile is { (x, y), (x+1, y) }
    # If (x + y) % 2 == 1, the tile is { (x-1, y), (x, y) }
    # So the "tile-x" coordinate is:
    # tx_coord = x // 2 if (x + y) % 2 == 0 else (x - 1) // 2
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's test: 
    # If y=0: (0,0)&(1,0) are tile 0, (2,0)&(3,0) are tile 1... -> tile_x = x // 2
    # If y=1: (1,1)&(2,1) are tile 0, (3,1)&(4,1) are tile 1... -> tile_x = (x-1) // 2
    
    # The cost to move from (sx, sy) to (tx, ty):
    # Every vertical step costs 1.
    # If we are at (sx, sy) and want to reach (tx, ty), we must cover |sy - ty| vertical distance.
    # This costs |sy - ty|.
    # After moving vertically, we are at (sx, ty). 
    # Now we need to move from (sx, ty) to (tx, ty).
    # If (sx, ty) and (tx, ty) are in the same tile, the cost is 0.
    # Otherwise, we need to move across tiles.
    # But we can optimize: we can move horizontally first, then vertically.
    # The most efficient way is to realize that the cost is:
    # |sy - ty| + (1 if the start and end are not in the same tile and we can't 
    #              reach the destination using only the vertical cost).
    
    # Actually, the simplest derivation for this specific tile pattern:
    # The cost is |sy - ty| + 1 if (sx + sy) % 2 != (tx + ty) % 2 
    # AND we are not in the same tile.
    # Let's use the property: 
    # A point (x, y) is in tile ( (x if (x+y)%2==0 else x-1)//2, y )
    # The distance is |sy - ty| + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)
    # Let's check Sample 1: 5 0 to 2 5.
    # |0 - 5| + (1 if (5+0)%2 != (2+5)%2 else 0) = 5 + (1 if 1 != 1 else 0) = 5. Correct.
    # Sample 2: 3 1 to 4 1.
    # |1 - 1| + (1 if (3+1)%2 != (4+1)%2 else 0) = 0 + (1 if 0 != 1 else 0) = 1. 
    # Wait, Sample 2 output is 0. Let's re-evaluate.
    # In Sample 2: (3,1) and (4,1). i=3, j=1. i+j = 4 (even).
    # Rule: A_{3,1} and A_{4,1} are in the same tile.
    # So (3.5, 1.5) and (4.5, 1.5) are in the same tile. Cost 0.
    # My formula: (3+1)%2 = 0, (4+1)%2 = 1. They are different.
    # The rule is: if (x + y) % 2 == 0, then x and x+1 are in the same tile.
    # So (x, y) and (x+1, y) are in the same tile if (x + y) % 2 == 0.
    
    # Correct logic:
    # To move from (sx, sy) to (tx, ty):
    # The vertical distance is dy = |sy - ty|.
    # Each vertical move costs 1.
    # We can move horizontally for free if we are within a tile.
    # The only way to avoid an extra cost is if we can reach the target 
    # using only the vertical moves and the free horizontal moves.
    # A move from (sx, sy) to (tx, ty) costs dy + 1 if 
    # (sx + sy) % 2 != (tx + ty) % 2 is NOT the condition.
    # The condition is: can we reach (tx, ty) from (sx, sy) in dy steps?
    # In each vertical step, the parity of (x + y) changes.
    # After dy steps, the parity of (x + y) changes by dy % 2.
    # So the parity of (sx + sy) becomes (sx + sy + dy) % 2.
    # We can reach (tx, ty) for free horizontally if (tx, ty) is in the same tile 
    # as (sx, ty + dy) or (sx, ty - dy).
    # (sx, ty) and (tx, ty) are in the same tile if:
    # 1. sx == tx
    # 2. (sx < tx and (sx + ty) % 2 == 0 and tx == sx + 1)
    # 3. (sx > tx and (tx + ty) % 2 == 0 and sx == tx + 1)
    
    # Actually, the simplest observation:
    # The cost is |sy - ty| + 1 if (sx + sy) % 2 != (tx + ty) % 2 else |sy - ty|.
    # Let's check Sample 1: (5+0)%2 = 1, (2+5)%2 = 1. Cost = |0-5| + 0 = 5.
    # Sample 2: (3+1)%2 = 0, (4+1)%2 = 1. Cost = |1-1| + 1 = 1. 
    # Still getting 1 for Sample 2. Let me re-read.
    # "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # Sample 2: S=(3,1), T=(4,1). i=3, j=1. i+j = 4 (even).
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    # My parity check: (3+1)%2 = 0. (4+1)%2 = 1.
    # If (sx + sy) % 2 == 0, then (sx, sy) and (sx+1, sy) are the same tile.
    # So if sx=3, sy=1, then (3,1) and (4,1) are the same tile.
    # This means if (sx + sy) % 2 == 0, we can move to sx+1 for free.
    # If (sx + sy) % 2 == 1, we can move to sx-1 for free.
    
    # Let's use the tile-id logic:
    # For a point (x, y), it belongs to tile:
    # If (x + y) % 2 == 0: tile_x = x // 2 if y%2==0 else (x+1)//2? No.
    # Let's use the rule: (i, j) and (i+1, j) are same if i+j is even.
    # This means for a fixed j:
    # If j is even: (0,j)&(1,j), (2,j)&(3,j)... are tiles.
    # If j is odd: (1,j)&(2,j), (3,j)&(4,j)... are tiles.
    # In both cases, (x, j) and (x', j) are in the same tile if:
    # (x + j) % 2 == 0 and x' == x + 1
    # OR (x + j) % 2 == 1 and x' == x - 1
    # This is equivalent to saying they are in the same tile if:
    # floor((x + (j % 2)) / 2) == floor((x' + (j % 2)) / 2)
    
    # Let's check Sample 2: sx=3, sy=1, tx=4, ty=1.
    # j = 1. (3 + 1)//2 = 2. (4 + 1)//2 = 2. Same!
    # Sample 1: sx=5, sy=0, tx=2, ty=5.
    # j=0: (5+0)//2 = 2. j=5: (2+1)//2 = 1.
    # The cost is |sy - ty| + (1 if tile_x(sx, sy) != tile_x(tx, ty) after vertical move).
    # Wait, the vertical move also changes the tile.
    # The most robust way:
    # The cost is |sy - ty| + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0).
    # Let's re-verify Sample 2: (3+1)%2 = 0, (4+1)%2 = 1. 
    # Still 1. There must be a flaw in my parity logic.
    # Let's use the tile-x formula: tx(x, y) = (x + (y % 2)) // 2.
    # Sample 2: tx(3, 1) = (3+1)//2 = 2. tx(4, 1) = (4+1)//2 = 2.
    # Since tx(3,1) == tx(4,1), the cost is |1-1| = 0.
    # Sample 1: tx(5, 0) = (5+0)//2 = 2. tx(2, 5) = (2+1)//2 = 1.
    # Cost: |0-5| + (1 if 2 != 1 else 0) = 5 + 1 = 6? No, Sample 1 is 5.
    # If we move vertically first: (5, 0) -> (5, 5).
    # tx(5, 5) = (5+1)//2 = 3. tx(2, 5) = (2+1)//2 = 1.
    # Then we move horizontally from tx=3 to tx=1.
    # The cost to move from tx_start to tx_end is |tx_start - tx_end|.
    # Total cost = |sy - ty| + |tx(sx, sy + (ty-sy