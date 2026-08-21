The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing the transformation of data. For this problem, the goal is to calculate the minimum toll between two points in a grid of $2\times1$ tiles. The optimal strategy involves moving diagonally (using a combination of horizontal and vertical steps) to minimize the number of tile boundaries crossed. I will use `abs()` for distances and `max()`/`min()` to implement the cost logic in a single expression.

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
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, tiles are paired (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    # Effectively, the "boundary" between tiles in the x-direction 
    # occurs at x = k when k % 2 == j % 2.
    
    # Let's transform the coordinates to a simpler system.
    # In the original system, moving vertically always crosses a tile boundary
    # unless you stay within the same 2x1 tile (which is impossible for y-movement).
    # Moving horizontally might not cross a boundary.
    
    # The cost to move from (sx, sy) to (tx, ty) can be viewed as:
    # Each vertical step costs 1.
    # Each horizontal step costs 1 ONLY if it crosses a boundary.
    # However, we can move diagonally. A move of (+1, +1) might cost 
    # only 1 if the horizontal component is "free".
    
    # Let's redefine coordinates:
    # A point (x, y) is in a tile. The tile ID can be represented as:
    # If (x+y) is even, the tile is { (x,y), (x+1,y) }.
    # We can map (x, y) to a new coordinate system (u, v) where:
    # v = y
    # u = (x if (x+y)%2 == 0 else x-1) // 2  <-- This is complex.
    
    # Simpler observation:
    # The distance is based on the Manhattan distance in a transformed grid.
    # Let x' = x, y' = y.
    # The cost is max(|sx-tx|, |sy-ty|) if we can move diagonally.
    # But the tiles are 2x1.
    # The distance is actually:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The cost is (dx + dy) // 2 if we optimize, but the parity of the 
    # tile boundaries matters.
    
    # Correct logic for this specific tiling:
    # The distance is max(|sx-tx|, |sy-ty|) if we consider the 
    # "cheapest" paths. Specifically, the cost is:
    # ceil( (abs(sx-tx) + abs(sy-ty)) / 2 ) 
    # BUT we must account for the starting and ending tile offsets.
    
    # Let's use the transformation:
    # X = x + y, Y = x - y
    # This is a standard trick for Manhattan distance, but here 
    # the tiles are 2x1.
    
    # The actual minimum cost is:
    # cost = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Wait, the simplest form for this specific problem is:
    # cost = (abs(sx - tx) + abs(sy - ty) + 1) // 2
    # Let's check Sample 1: 5 0 to 2 5 -> dx=3, dy=5. (3+5+1)//2 = 4. Incorrect (Sample says 5).
    
    # Re-evaluating:
    # In row y, boundaries are at x = k where k % 2 == y % 2.
    # This is like a brick wall.
    # The distance between (sx, sy) and (tx, ty) in a brick wall is:
    # dist = max(|sy - ty|, (|sx - tx| + 1) // 2 + (some offset))
    # Actually, the distance is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: dx=3, dy=5. cost = 5 + max(0, (3-5+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1 -> dx=1, dy=0. cost = 0 + max(0, (1-0+1)//2) = 1. 
    # Wait, Sample 2 says 0. 
    # In Sample 2: (3,1) and (4,1). i=3, j=1. i+j = 4 (even).
    # Rule: A_{i,j} and A_{i+1,j} are same tile.
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    
    # The condition for A_{x,y} and A_{x+1,y} being the same tile is (x+y) % 2 == 0.
    # Let's normalize x based on y:
    # If (x+y) % 2 != 0, then x is the "right" half of a tile.
    # We can think of the tiles as being indexed by (x', y) where x' = (x if (x+y)%2==0 else x-1) // 2
    # This is still confusing. Let's use:
    # A tile is defined by (y, (x + (y%2)) // 2)
    # Let nx = (sx + (sy % 2)) // 2, ny = sy
    # Let mx = (tx + (ty % 2)) // 2, my = ty
    # The distance is then the Manhattan distance in the (nx, ny) space?
    # No, because moving in y might change the x-coordinate of the tile.
    
    # Correct approach:
    # The cost is max(|sy - ty|, (abs(sx - tx) + (1 if (sx+sy)%2 != 0 else 0) 
    #                            - (1 if (tx+ty)%2 == 0 else 0) ... ))
    # Actually, the simplest formula for this grid is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: dx=1, dy=0. max(0, (1+0+1)//2) = 1. Still 1.
    
    # Let's use the coordinate transformation:
    # Each tile can be identified by (y, x_tile) where x_tile = (x + (y%2)) // 2
    # Start: (sy, (sx + (sy%2)) // 2), End: (ty, (tx + (ty%2)) // 2)
    # In one move, you can change y by n (cost n) and x_tile by 0.
    # Or change x_tile by n (cost n) and y by 0.
    # But you can also move diagonally: (y+1, x_tile+1) or (y+1, x_tile-1).
    # A move from (y, x_tile) to (y+1, x_tile') costs 1.
    # x_tile' can be x_tile or x_tile - 1 (because the offset y%2 flips).
    # This is exactly the distance in a grid where you can move to 
    # (y+/-1, x_tile) and (y+/-1, x_tile-1) and (y, x_tile+/-1).
    # This is the Chebyshev distance if we shift the coordinates.
    # The distance is max(|sy - ty|, |(sx + (sy%2))//2 - (tx + (ty%2))//2|)
    # Wait, the "diagonal" move is: from tile (y, x_t) to (y+1, x_t) 
    # costs 1 and covers 1 unit of y and potentially 1 unit of x.
    # The distance is max(abs(sy - ty), abs((sx + (sy%2))//2 - (tx + (ty%2))//2))
    # Sample 1: sx=5, sy=0 -> (0, (5+0)//2) = (0, 2). tx=2, ty=5 -> (5, (2+1)//2) = (5, 1).
    # max(|0-5|, |2-1|) = 5. Correct.
    # Sample 2: sx=3, sy=1 -> (1, (3+1)//2) = (1, 2). tx=4, ty=1 -> (1, (4+1)//2) = (1, 2).
    # max(|1-1|, |2-2|) = 0. Correct.
    # Sample 3: sx=2552608206527595, sy=5411232866732612 
    # tx=771856005518028, ty=7206210729152763
    # sy%2 = 0, ty%2 = 1
    # nx = (2552608206527595 + 0) // 2 = 1276304103263797
    # ny = 5411232866732612
    # mx = (771856005518028 + 1) // 2 = 385928002759014
    # my = 7206210729152763
    # max(|5411232866732612 - 7206210729152763|, |1276304103263797 - 385928002759014|)
    # max(794977862420151, 890376100504783) = 890376100504783.
    # Sample 3 output is 1794977862420151. My formula is wrong.
    
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # This is distance in the dual graph.
    # The distance is simply the Manhattan distance in the (x_tile, y) grid?
    # Let's check Sample 3 with Manhattan:
    # abs(5411232866732612 - 7206210729152763) + abs(1276304103263797 - 385928002759014)
    # = 794977862420151 + 890376100504783 = 1685353962924934. Still not 1794977862420151.
    
    # Let's use the property: cost is abs(sy - ty) + abs(nx - mx)
    # But wait, the x_tile depends on y.
    # When moving from y to y+1, the tile boundaries shift.
    # The distance is actually:
    # cost = abs(sy - ty) + abs((sx + (sy%2))//2 - (tx + (ty%2))//2)
    # Let's re-calculate Sample 3: 79497786242