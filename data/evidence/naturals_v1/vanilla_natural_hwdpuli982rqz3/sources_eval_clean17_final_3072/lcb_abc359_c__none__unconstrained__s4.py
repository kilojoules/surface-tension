```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern is defined such that A_{i,j} and A_{i+1,j} are in the same tile if i+j is even.
    # This means for a fixed j, the tiles are horizontal 2x1 blocks starting at (i, j) where i+j is even.
    # Let's transform the coordinates to a system where the distance is easier to calculate.
    # The cost to move between two points in this specific grid layout can be derived
    # by observing that moving vertically always crosses a tile boundary (cost 1),
    # and moving horizontally crosses a boundary every 2 units, but the boundary shifts.
    
    # A known approach for this specific tiling problem (often found in competitive programming)
    # is to transform the coordinates (x, y) to (u, v) where:
    # u = x + y
    # v = x - y
    # However, a more direct way to think about it is:
    # To move from (sx, sy) to (tx, ty), the minimum cost is:
    # ceil( abs(sx - tx) / 2 ) + abs(sy - ty)
    # But this depends on the parity of the coordinates.
    
    # Let's use the property:
    # Cost = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Actually, for this specific problem layout:
    # The distance is abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2)
    # Wait, the simplest correct formula for this specific grid is:
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # The cost is dy + max(0, (dx + 1) // 2 - (dy + 1) // 2) if we align them.
    # Actually, the most robust formula for this problem is:
    # cost = max(dy, (dx + dy + 1) // 2)
    # Let's verify with Sample 1: 5 0, 2 5 -> dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: 3 1, 4 1 -> dx=1, dy=0. max(0, (1+0+1)//2) = 1? No, Sample 2 is 0.
    
    # Let's re-evaluate. 
    # In row j, tiles are [0,1], [2,3] if j is even, and [1,2], [3,4] if j is odd.
    # To move from (sx, sy) to (tx, ty):
    # If we move only vertically, cost is abs(sy - ty).
    # If we move horizontally, we can "skip" every other boundary.
    # The correct logic:
    # To get from sx to tx, we need to cross boundaries.
    # In a row j, the boundaries are at x = k where k % 2 != j % 2.
    # The number of boundaries between sx and tx in row j is:
    # count(k | sx < k <= tx and k % 2 != j % 2)
    
    # Let's use the coordinate transformation:
    # Let X = sx + sy, Y = sx - sy
    # Let X' = tx + ty, Y' = tx - ty
    # The distance is max(abs(X - X') // 2, abs(Y - Y') // 2)
    # This is a known property for this specific L1-like grid with 2x1 tiles.
    
    # Let's test Sample 1: (5,0) -> (2,5). X=5, Y=5; X'=7, Y'=-3.
    # max(abs(5-7)//2, abs(5 - -3)//2) = max(1, 4) = 4. Still not 5.
    
    # Let's use the logic: 
    # To move from (sx, sy) to (tx, ty), you must change your 'y' coordinate abs(sy-ty) times.
    # Each such move costs 1.
    # Additionally, you might need to move horizontally.
    # The horizontal boundaries are at x = k where k % 2 != y % 2.
    # If you are at (sx, sy) and move to (tx, sy), you cross boundaries.
    # Then you move to (tx, ty).
    # Total cost = abs(sy - ty) + (number of boundaries crossed horizontally that aren't "covered" by vertical moves).
    
    # Correct mathematical derivation for this problem:
    # The cost is abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2)
    # Wait, Sample 2: 3 1, 4 1 -> dx=1, dy=0. 0 + max(0, 1 - 0) = 1. Still not 0.
    # Sample 2: (3,1) and (4,1). i=3, j=1. i+j=4 (even). 
    # A_{3,1} and A_{4,1} are in the same tile. So cost is 0.
    
    # Let's use the property:
    # Two cells (i, j) and (i', j') are in the same tile if j=j' and (i+j)%2 == 0 and i' = i+1 (or vice versa).
    # This is a shortest path problem on a graph.
    # The distance is:
    # dist = abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2)
    # No, the simplest form is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: max(5, (3+5+1)//2) = 5.
    # Sample 2: max(0, (1+0+1)//2) = 1. Still 1.
    
    # Let's reconsider: if we are at (sx, sy), we are in tile T1.
    # T1 is { (sx, sy), (sx+1, sy) } if sx+sy is even, else { (sx-1, sy), (sx, sy) }.
    # Let's normalize: every tile is {(2k, j), (2k+1, j)} where k is some integer.
    # To do this, if j is odd, we shift the x-coordinates by 1.
    # Let f(x, y) = (x + (y % 2), y)
    # The distance between (sx, sy) and (tx, ty) in the original grid is the 
    # Manhattan distance between f(sx, sy) and f(tx, ty) divided by 2?
    # Let's try: f(5, 0) = (5, 0), f(2, 5) = (3, 5).
    # dx = 2, dy = 5. Cost = (2 + 5 + 1) // 2 = 4.
    
    # Final attempt at logic:
    # The cost is abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (abs(sy - ty) + 1) // 2)
    # But we must account for the parity of sx+sy and tx+ty.
    # Let's use the coordinate transform: u = x, v = y.
    # If we move from (x, y) to (x, y+1), cost 1.
    # If we move from (x, y) to (x+1, y), cost 1 UNLESS (x+y) is even.
    # This is equivalent to:
    # Cost = abs(sy - ty) + (number of x-boundaries crossed)
    # To minimize this, we can use the vertical moves to "skip" x-boundaries.
    # Each vertical move from y to y+1 allows us to cross one x-boundary for free 
    # because the boundary positions shift.
    
    # The minimum cost is actually:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # result = max(dy, (dx + dy + 1) // 2)
    # But we must check if the start and end points are in the same tile.
    # Sample 2: (3,1) and (4,1). 3+1=4 (even). They are in the same tile. Cost 0.
    # My formula gives max(0, (1+0+1)//2) = 1.
    # The formula max(dy, (dx + dy + 1) // 2) assumes we start at a boundary.
    # If we start inside a tile, we can move to the boundary for free.
    # Let's adjust sx, tx to the boundaries of the tiles they belong to.
    # Tile for (sx, sy): if (sx+sy)%2 == 0, tile is [sx, sx+1]. If (sx+sy)%2 == 1, tile is [sx-1, sx].
    # Let L_s = sx if (sx+sy)%2 == 0 else sx-1.
    # Let L_t = tx if (tx+ty)%2 == 0 else tx-1.
    # Now we move from L_s to L_t.
    # dx = abs(L_s - L_t) / 2 (since these are now indices of 2x1 tiles)
    # dy = abs(sy - ty)
    # Cost = max(dy, (dx + dy + 1) // 2) ... no.
    
    # Let's use the most reliable formula for this problem:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # To handle the "inside tile" part:
    # If (sx + sy) % 2 == 0, we can effectively treat sx as sx + 1 for free.
    # If (tx + ty) % 2 == 0, we can effectively treat tx as tx - 1 for free.
    # This is getting confusing. Let's use:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # The answer is (dx + dy + 1) // 2 if dx > dy else dy.
    # Wait, Sample 2: dx=1, dy=0 -> (1+0+1)//2 = 1. Still 1.
    # The only way Sample 2 is 0 is if dx=1, dy=0 and they are in the same tile.
    # If dx=0, dy=0, cost 0.
    # If dx=1, dy=0, cost 0 if (min(sx, tx) + sy) % 2 == 0, else 1.
    # If dx=0, dy=1, cost 1.
    
    # Correct logic:
    # Let's shift coordinates so that all tiles are {(2k, j), (2k+1, j)}.
    # To do this, if j is odd, x becomes x-1.
    # New coordinates: (sx', sy) = (sx if sy%2==0 else sx-1, sy)
    # (tx', ty) = (tx if ty%2==0 else tx-1, ty)
    # Now, horizontal boundaries are at every even x.
    # To move from sx' to tx', we cross abs(sx' - tx') / 2 boundaries.
    # Each vertical move can "bypass" one horizontal boundary.
    # Cost = abs(sy - ty) + max(0, abs(sx' - tx') // 2 - abs(sy - ty))
    # Let's test Sample 1: (5,0) -> (2,5).
    # f(5,0) = (5,0), f(2,5) = (1,5).
    # dy = 5, dx' = 4. Cost = 5 + max(0, 2 - 5) = 5. Correct.
    # Sample 2: (3,1) -> (4,1).
    # f(3,1) = (2,1), f(4,1) =