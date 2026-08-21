The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing expressions over statements. For this specific problem, since the optimal path is determined by a mathematical formula based on the coordinates, I will use a direct calculation within a `print` statement, utilizing `abs` for distances and integer division to determine the number of tile boundaries crossed.

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
    # When i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j, tiles are paired as (0,1), (2,3), (4,5)... if j is even
    # and (-1,0), (1,2), (3,4)... if j is odd.
    # Essentially, a tile is defined by (floor(x), j) and (floor(x)+1, j) 
    # if (floor(x) + j) is even.
    
    # Let's define the tile index for a cell (i, j):
    # If (i + j) is even, the tile is T_{i, j} = {A_{i,j}, A_{i+1,j}}
    # If (i + j) is odd, the tile is T_{i, j} = {A_{i-1,j}, A_{i,j}}
    # Notice that for any (i, j), it belongs to a tile that spans x-coordinates 
    # {2k, 2k+1} if j is even, or {2k-1, 2k} if j is odd.
    
    # The cost to move between two points (sx, sy) and (tx, ty):
    # Moving vertically across j-boundaries always costs 1 per unit.
    # Moving horizontally: if we are in the same tile, cost is 0.
    # The tiles are shifted every row. 
    # The distance is effectively the L1 distance, but we can "skip" 
    # horizontal costs if we align with the tile boundaries.
    
    # The minimum cost is:
    # cost = abs(sy - ty) + (distance in x that cannot be covered by the 
    # tiles we are already paying for via vertical movement).
    
    # More simply: in each row j, we are in a tile of width 2.
    # The total cost is abs(sy - ty) + max(0, abs(sx - tx) - (1 if sy%2 != ty%2 else 0))
    # Wait, that's for 1D. For 2D:
    # The optimal strategy is to move vertically and use the "free" horizontal 
    # step provided by the tile width.
    # Each vertical step (change in j) allows us to move 1 unit horizontally for free
    # because the tile boundaries shift.
    
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # We must pay dy for vertical moves.
    # Each vertical move allows us to effectively move 1 unit horizontally 
    # (by picking the correct tile).
    # However, the tiles are fixed. 
    # Let's use the property: cost = max(dy, (dx + dy + 1) // 2) if we can move diagonally.
    # But we can only move L, R, U, D.
    # The actual minimum cost is:
    # If we move dy units vertically, we can cover some dx.
    # The parity of (i+j) determines the tile.
    # The cost is dy + max(0, (dx - dy + 1) // 2) if we optimize.
    # Actually, the simplest form is: cost = (dx + dy + 1) // 2 
    # if we can utilize the tiles. But we must pay for every vertical boundary.
    # The correct formula for this specific tiling is:
    # cost = dy + max(0, (dx - dy + 1) // 2) if we can't move diagonally.
    # Let's re-evaluate: 
    # To move dx horizontally, we need dx//2 tiles if we are aligned.
    # If we move dy vertically, we cross dy boundaries.
    # The minimum cost is max(dy, (dx + dy + 1) // 2).
    # Let's check Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+0+1)//2) = 1? 
    # Wait, Sample 2 says 0. 
    # In Sample 2: (3,1) and (4,1). i=3, j=1. i+j = 4 (even).
    # Rule: if i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    # My formula max(0, (1+0+1)//2) gave 1. The issue is the parity of (sx+sy).
    
    # Correct logic:
    # Let's normalize so sx <= tx and sy <= ty.
    # The cost is dy + (remaining dx after using the "free" steps from dy).
    # Each vertical step from j to j+1 allows us to move from tile T_{i,j} to T_{i,j+1}.
    # Since the tiles shift, we can cover 1 unit of dx for every 1 unit of dy.
    # The remaining dx is dx - dy. If this is > 0, we need (dx - dy + 1) // 2 more tiles.
    # But we must account for the starting tile.
    # If (sx + sy) is even, we are in a tile covering {sx, sx+1}.
    # If (sx + sy) is odd, we are in a tile covering {sx-1, sx}.
    
    # Let's use the property:
    # The cost is dy + max(0, (dx - dy + (1 if (sx+sy)%2 == (tx+ty)%2 and dx > dy else 0)) // 2)
    # This is getting complex. Let's use the simplest observation:
    # The cost is max(dy, (dx + dy + 1) // 2) is almost correct, 
    # but the "free" horizontal step depends on the parity of the start.
    # If we move dy steps, we can cover dx steps if dx <= dy + 1 (depending on parity).
    # The most robust formula:
    # cost = dy + max(0, (dx - (dy + 1) + 1) // 2) if we start in a "good" tile.
    # Actually: cost = max(dy, (dx + dy + 1) // 2) is correct if we can start 
    # and end in the "middle" of the tiles.
    # Let's test Sample 2 again: dx=1, dy=0. (1+0+1)//2 = 1. Still 1.
    # The reason Sample 2 is 0 is because (3,1) and (4,1) are in the same tile.
    # (3+1) is even, so A_{3,1} and A_{4,1} are one tile.
    # If we were at (2,1) and (3,1), 2+1=3 (odd), so A_{2,1} and A_{3,1} are one tile.
    # In both cases, dx=1, dy=0 results in cost 0.
    # So if dy=0, cost is 0 if (sx, sy) and (tx, sy) are in the same tile, else 1.
    # They are in the same tile if (min(sx, tx) + sy) is even and abs(sx-tx) == 1.
    
    # Final logic:
    # The cost is dy + max(0, (dx - dy - 1 + 1) // 2) if we can use the 
    # start and end tiles effectively.
    # The most reliable formula for this problem is:
    # cost = max(dy, (dx + dy + 1) // 2) is for a different tiling.
    # For this tiling: cost = dy + max(0, (dx - dy - 1 + 1) // 2) 
    # is still not quite right.
    # Let's use: cost = dy + max(0, (dx - dy + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # No. Let's use:
    # The distance is dy + max(0, (dx - dy + 1) // 2) if we are "unlucky" with parity.
    # Actually, the simplest correct formula is:
    # cost = max(dy, (dx + dy + 1) // 2) if we consider the tiles as 
    # nodes in a graph. But the parity of sx+sy matters.
    # If (sx+sy)%2 == 0, we are in tile {sx, sx+1}.
    # If (sx+sy)%2 == 1, we are in tile {sx-1, sx}.
    # Let's adjust sx and tx to the "left" edge of their tiles.
    # Start tile left edge: Ls = sx if (sx+sy)%2 == 0 else sx-1
    # End tile left edge: Lt = tx if (tx+ty)%2 == 0 else tx-1
    # Now we want to move from Ls to Lt.
    # The distance is dy + max(0, (abs(Ls - Lt) + 1) // 2)
    # Wait, if we move dy, we can change the parity of the row.
    # The most direct formula:
    # cost = dy + max(0, (abs(sx - tx) - dy + 1) // 2)
    # Let's check Sample 1: 5,0 to 2,5. dx=3, dy=5. 5 + max(0, (3-5+1)//2) = 5.
    # Sample 2: 3,1 to 4,1. dx=1, dy=0. 0 + max(0, (1-0+1)//2) = 1. Still 1!
    # What is wrong? In Sample 2, (3,1) and (4,1) are in the same tile.
    # (3+1) is even, so A_{3,1} and A_{4,1} are one tile.
    # My Ls, Lt logic: Ls = 3 (since 3+1=4 even), Lt = 4 (since 4+1=5 odd, Lt=4-1=3).
    # Ls=3, Lt=3. Cost = 0 + max(0, (3-3+1)//2) = 0. Correct!
    
    # Final Formula:
    # Ls = sx if (sx + sy) % 2 == 0 else sx - 1
    # Lt = tx if (tx + ty) % 2 == 0 else tx - 1
    # cost = abs(sy - ty) + max(0, (abs(Ls - Lt) + 1) // 2)
    # Wait, the (abs(Ls-Lt)+1)//2 is only if we can't use the vertical steps.
    # If we move dy vertically, we can cover some dx.
    # Each vertical step allows us to move 1 unit of dx for free.
    # So we can cover dx_covered = dy units of dx.
    # The remaining dx is dx_rem = abs(Ls - Lt) - dy.
    # The cost is dy + max(0, (dx_rem + 1) // 2).
    
    # Let's refine:
    # 1. Find the left-edge of the tile containing (sx, sy): Ls
    # 2. Find the left-edge of the tile containing (tx, ty): Lt
    # 3. dx_tiles = abs(Ls - Lt) // 2
    # 4. The cost is max(abs(sy - ty), dx_tiles)
    # Let's check Sample 1: Ls = 5 (5+0=5 odd, so 5-1=4), Lt = 2 (2+5=7 odd, so 2-1=1).
    # dx_tiles = abs(4 - 1) // 2 = 3 // 2