The constraint to avoid explicit loops and recursion forces a shift toward functional programming paradigms, emphasizing declarative data transformations over imperative state changes. For this problem, which requires calculating a distance-like metric on a tiled grid, I will use direct arithmetic expressions to compute the cost based on the parity of coordinates and the Manhattan distance, as the optimal path depends on how many tile boundaries are crossed.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling pattern:
    # If i+j is even, (i, j) and (i+1, j) are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, if j is even, tiles are [0,1], [2,3], ...
    # In row j, if j is odd, tiles are [-1,0], [1,2], ...
    
    # Let's normalize the coordinates to a grid of tiles.
    # For a fixed y, the tile index x_tile is:
    # If y is even: x_tile = x // 2
    # If y is odd:  x_tile = (x - 1) // 2 if x >= 0 else (x - 1) // 2
    # Actually, a simpler way: if (x + y) % 2 == 0, x is the left half of a tile.
    # The tile boundaries are at x = k when (k + y) is odd.
    
    # The cost to move between (sx, sy) and (tx, ty):
    # Moving vertically always crosses a tile boundary unless you stay in the same tile.
    # Moving horizontally might not cross a boundary if you stay within the 2x1 block.
    
    # Let's transform the coordinates to a coordinate system where each unit is a tile.
    # In the original system, a tile is 2x1.
    # Let's use the transformation:
    # X = x + (y % 2)
    # Y = y
    # Then the tiles are aligned as 2x1 blocks starting at even X.
    # The cost is related to the Manhattan distance in a transformed space.
    
    # Specifically, the cost to move from (sx, sy) to (tx, ty) is:
    # cost = max(0, (abs(sx - tx) + abs(sy - ty) + 1) // 2) is too simple.
    
    # Correct logic:
    # Let's define a coordinate transformation:
    # u = x + y
    # v = x - y
    # This is a common trick for Manhattan distances on grids, but here the tiles are 2x1.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # Each vertical step (y -> y+1) always enters a new tile.
    # Each horizontal step (x -> x+1) enters a new tile UNLESS (x+y) is even and we move to x+1.
    
    # The minimum cost is:
    # abs(sy - ty) + (cost to move from sx to tx given the parity of y)
    # However, we can move diagonally by combining a horizontal and vertical move.
    # The optimal strategy is to move in a "zigzag" fashion.
    
    # The distance is:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # The cost is (dx + dy) // 2 if we can utilize the 2x1 tiles.
    # Specifically, the distance is ceil((dx + dy) / 2) 
    # BUT we must account for the starting and ending tile alignments.
    
    # Let's use the property:
    # A move (x, y) -> (x+1, y) costs 0 if (x+y) is even, 1 if (x+y) is odd.
    # A move (x, y) -> (x, y+1) always costs 1.
    
    # This is equivalent to a graph where edges have weights 0 or 1.
    # The distance is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Wait, if we move vertically, we can change the parity of (x+y) to make horizontal moves free.
    # The cost is actually:
    # If we move dy vertically, we can "absorb" some dx.
    # Each vertical move allows us to potentially cross a 0-cost horizontal boundary.
    
    # The formula for this specific tiling problem is:
    # cost = (abs(sx - tx) + abs(sy - ty) + 1) // 2
    # But we must check if the start and end points are in the same tile.
    
    # Let's refine:
    # The distance is:
    # d = abs(sy - ty)
    # remaining_x = abs(sx - tx)
    # If we move d steps vertically, we can cover some horizontal distance.
    # The cost is max(d, (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Actually, the simplest correct form for this problem is:
    # ans = (abs(sx - tx) + abs(sy - ty) + 1) // 2
    # But we must subtract 1 if (sx, sy) and (tx, ty) are in the same tile.
    # Wait, the sample 1: (5,0) to (2,5). dx=3, dy=5. (3+5+1)//2 = 4. Sample output is 5.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0. (1+0+1)//2 = 1. Sample output is 0.
    
    # Let's re-evaluate.
    # In row y, tiles are [x, x+1] if x+y is even.
    # Start: (sx, sy). End: (tx, ty).
    # Let's use the transformation:
    # X = x, Y = y
    # Cost to move (x, y) -> (x+1, y) is 0 if x+y even, 1 if x+y odd.
    # Cost to move (x, y) -> (x, y+1) is 1.
    
    # This is a shortest path problem on a graph.
    # The distance is:
    # dist = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5.
    # dx=3, dy=5. dist = 5 + max(0, (3-5+1)//2) = 5 + 0 = 5. Correct.
    # Sample 2: sx=3, sy=1, tx=4, ty=1.
    # dx=1, dy=0. dist = 0 + max(0, (1-0+1)//2) = 1. 
    # Wait, Sample 2 output is 0. Why?
    # (3,1): 3+1=4 (even). So A_{3,1} and A_{4,1} are the same tile.
    # Since (3.5, 1.5) and (4.5, 1.5) are in the same tile, cost is 0.
    
    # The condition for 0 cost is: sy == ty and (sx + sy) % 2 == 0 and tx == sx + 1
    # Or sy == ty and (tx + sy) % 2 == 0 and sx == tx + 1.
    
    # Let's use the general formula:
    # The cost is the distance in a graph.
    # The nodes are tiles.
    # Tile ID for (x, y):
    # If (x + y) % 2 == 0: tile_id = ( (x // 2 if y % 2 == 0 else (x-1) // 2), y )
    # This is getting complex. Let's use the property:
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # But we must adjust for the starting and ending tile offsets.
    
    # Let's use the coordinate transformation:
    # x' = x + (y % 2)
    # Then tiles are 2x1 blocks starting at even x'.
    # The distance is:
    # dx = abs(sx + (sy % 2) - (tx + (ty % 2)))
    # dy = abs(sy - ty)
    # The cost is dy + max(0, (dx // 2)) if we are careful.
    
    # Let's try:
    # x1 = sx + (sy % 2)
    # x2 = tx + (ty % 2)
    # cost = abs(sy - ty) + (abs(x1 - x2) + 1) // 2
    # Sample 1: sx=5, sy=0, tx=2, ty=5.
    # x1 = 5 + 0 = 5; x2 = 2 + 1 = 3.
    # cost = 5 + (abs(5-3)+1)//2 = 5 + 1 = 6. Still wrong.
    
    # Let's try:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # For Sample 2: sx=3, sy=1, tx=4, ty=1.
    # dx=1, dy=0. cost = 0 + (1-0+1)//2 = 1.
    # But if (sx, sy) and (tx, ty) are in the same tile, cost is 0.
    # They are in the same tile if sy == ty and (min(sx, tx) + sy) % 2 == 0 and abs(sx - tx) == 1.
    
    # Actually, the most reliable way to solve this is to recognize it as a 
    # distance on a grid where some edges are 0.
    # The cost is:
    # ans = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Then we subtract 1 if the start and end are in the same tile.
    # But that's only if the path was just one step.
    
    # Let's use the logic:
    # Each vertical move costs 1 and can "cover" 1 unit of horizontal distance for free.
    # If we have dx and dy:
    # We use dy vertical moves to cover min(dx, dy) horizontal distance.
    # The remaining horizontal distance is max(0, dx - dy).
    # This remaining distance costs 1 per 2 units (since tiles are 2x1).
    # Cost = dy + (max(0, dx - dy) + 1) // 2
    # Let's check Sample 1: dx=3, dy=5. Cost = 5 + (0+1)//2 = 5. Correct.
    # Sample 2: dx=1, dy=0. Cost = 0 + (1+1)//2 = 1. Still 1.
    # Wait, in Sample 2, (3,1) and (4,1) are in the same tile.
    # The cost to move between them is 0.
    # My formula gives 1 because it doesn't know WHICH tile they are in.
    
    # If sy == ty, the cost is 0 if (min(sx, tx) + sy) % 2 == 0 and abs(sx - tx) == 1.
    # Otherwise, if sy == ty, the cost is (abs(sx - tx) + 1) // 2 if we can't use the 0-edge.
    # Actually, if sy == ty, the cost is:
    # if sx == tx: 0
    # if (min(sx, tx) + sy) % 2 == 0 and abs(sx - tx) == 1: 0
    # else: (abs(sx - tx) + 1) // 2 ... no, that's not right.
    
    # Let's use the property:
    # The cost is 0 if they are in the same tile.
    # They are in the same tile if sy == ty and (sx + sy) % 2 == 0 and tx == sx + 