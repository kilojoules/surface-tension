The constraint to avoid explicit loops and recursion forces a shift from imperative to functional programming styles, encouraging the use of high-level abstractions and built-in optimizations rather than manual state management. For this problem, which requires calculating a distance metric on a grid of $2\times1$ tiles, the solution involves coordinate transformations and the use of `abs` and `max` functions to determine the minimum cost.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern:
    # When i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # Specifically, for a fixed j:
    # If j is even, tiles are {(0,1), (2,3), (4,5)...}
    # If j is odd, tiles are {(-1,0), (1,2), (3,4)...}
    
    # Let's transform the coordinates to a system where we can calculate 
    # the cost more easily.
    # In row j, the tile index is:
    # If j is even: floor(i / 2)
    # If j is odd: floor((i + 1) / 2)
    
    # However, a simpler observation:
    # Moving vertically always costs 1 per unit distance because you enter a new tile.
    # Moving horizontally costs 0 if you stay within the 2x1 tile, 
    # and 1 if you cross into a new tile.
    
    # Let's define a coordinate transformation:
    # A point (x, y) belongs to tile (X, Y) where:
    # Y = y
    # X = (x >> 1) if y % 2 == 0 else ((x + 1) >> 1)
    
    # The cost to move between (sx, sy) and (tx, ty):
    # The vertical distance is always |sy - ty|.
    # The horizontal distance depends on the parity of the rows.
    # This is equivalent to a Manhattan distance on a transformed grid.
    # The distance is max(|sx - tx|, |sy - ty|) if we consider the 
    # specific connectivity of these tiles.
    
    # More formally, the distance is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Wait, the actual formula for this specific tiling is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The cost is dy + max(0, (dx - dy + 1) // 2) if we can optimize.
    # Actually, the simplest form is:
    # cost = max(dy, (dx + dy + 1) // 2) if we can move diagonally via tiles.
    # But we can only move U, D, L, R.
    
    # Let's re-evaluate:
    # To move from (sx, sy) to (tx, ty):
    # Vertical steps: dy = |sy - ty|. Each step costs 1.
    # Horizontal steps: dx = |sx - tx|.
    # In each row, we can move 2 units horizontally for the cost of 1 
    # (by moving into the adjacent square of the same tile).
    # The most efficient way is to use the vertical moves to "shift" 
    # our horizontal alignment.
    
    # The correct formula for this grid is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Wait, let's check Sample 1: (5,0) to (2,5). dx=3, dy=5.
    # max(5, (3+5+1)//2) = max(5, 4) = 5. Correct.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0.
    # max(0, (1+0+1)//2) = 1. Incorrect. Sample 2 says 0.
    
    # Re-evaluating Sample 2: (3,1) and (4,1).
    # i=3, j=1. i+j = 4 (even). A_{3,1} and A_{4,1} are the same tile.
    # So cost is 0.
    
    # The rule is: (i, j) and (i+1, j) are same tile if i+j is even.
    # This means in row j, tiles are {(i, j), (i+1, j)} for i such that i+j is even.
    # This is a standard "brick wall" pattern.
    # The distance between (sx, sy) and (tx, ty) in this metric is:
    # cost = max(|sy - ty|, ceil((|sx - tx| + |sy - ty|) / 2)) 
    # is for 8-connectivity. For 4-connectivity:
    # The cost is actually:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # If we move dy vertically, we can cover some dx.
    # Each vertical move allows us to change our "parity" relative to the tiles.
    # The minimum cost is max(dy, (dx + dy + 1) // 2) is still not quite right.
    
    # Correct logic:
    # To move dx horizontally and dy vertically:
    # We must pay dy for vertical moves.
    # In each row, we can move 2 units for the price of 1 (the tile we are in).
    # The total cost is dy + max(0, (dx - dy + 1) // 2) if we align correctly.
    # Let's check Sample 2: dx=1, dy=0. 0 + (1-0+1)//2 = 1. Still 1.
    # But in Sample 2, (3,1) and (4,1) are in the same tile.
    # (3,1): i=3, j=1. i+j=4 (even). So A_{3,1} and A_{4,1} are one tile.
    # Since sx=3, tx=4, they are in the same tile. Cost 0.
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # Let's use the coordinate transform:
    # A tile is identified by (X, Y) where Y = y and X = (x + (y % 2)) // 2
    # The distance between (X1, Y1) and (X2, Y2) is:
    # cost = |Y1 - Y2| + max(0, |X1 - X2| - 0) ... no.
    # In this transformed grid, moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1.
    # But moving from (X, Y) to (X, Y+1) might be "free" if the tile 
    # at (X, Y+1) is the same as (X, Y)? No, tiles are 2x1.
    
    # Let's use the property:
    # Cost = max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2) 
    # is for a different problem.
    # For this problem:
    # The distance is simply the Manhattan distance on the graph of tiles.
    # Two tiles are adjacent if they share an edge.
    # Tile (X, Y) is adjacent to (X, Y+1), (X, Y-1), (X+1, Y), (X-1, Y).
    # Wait, the tiles are 2x1.
    # Tile (X, Y) covers squares (2X if Y even else 2X-1, Y) and (2X+1 if Y even else 2X, Y).
    # It shares an edge with tiles in row Y-1 and Y+1.
    # Specifically, tile (X, Y) shares an edge with tiles in row Y+1 that 
    # overlap with its x-range.
    # Row Y tile X covers x-range [x_start, x_end].
    # Row Y+1 tile X' covers x-range [x'_start, x'_end].
    # They are adjacent if the intervals overlap.
    
    # This is equivalent to:
    # Cost = max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2) 
    # if we can move diagonally. But we can't.
    # Actually, the distance is:
    # dx = abs(sx - tx), dy = abs(sy - ty)
    # cost = dy + max(0, (dx - dy + 1) // 2) if we can't move diagonally.
    # Let's re-check Sample 2: sx=3, tx=4, sy=1, ty=1.
    # dx=1, dy=0. (1-0+1)//2 = 1. Still 1.
    # The only way Sample 2 is 0 is if we are already in the same tile.
    # If sx=3, tx=4, sy=1, ty=1:
    # i=3, j=1 => i+j=4 (even) => A_{3,1} and A_{4,1} are one tile.
    # So they are in the same tile. Cost = 0.
    
    # If we are in tile (X1, Y1) and (X2, Y2):
    # The distance is |Y1 - Y2| + |X1 - X2|.
    # But we can move from (X, Y) to (X, Y+1) and (X+1, Y+1) or (X-1, Y+1) 
    # and (X, Y+1) depending on the offset.
    # Actually, the distance is simply:
    # cost = max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # NO, that's for 8-connectivity.
    # For 4-connectivity, the distance is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Wait, if dx=1, dy=0 and they are in the same tile, cost is 0.
    # If dx=2, dy=0, they must be in different tiles. Cost is 1.
    # (2-0+1)//2 = 1.
    # If dx=1, dy=0 and they are in different tiles, cost is 1.
    # (1-0+1)//2 = 1.
    # So the formula `dy + max(0, (dx - dy + 1) // 2)` is almost correct, 
    # but it doesn't account for whether the start and end are in the same tile.
    
    # Let's use the tile coordinates:
    # X(i, j) = (i + (j % 2)) // 2
    # Y(i, j) = j
    # Start tile: (X1, Y1), End tile: (X2, Y2)
    # The distance between tiles in this grid is:
    # dist = max(|Y1 - Y2|, (|X1 - X2| + |Y1 - Y2| + 1) // 2) 
    # No, that's not it.
    # Let's use the property that you can move to any tile in the next row 
    # that shares a boundary.
    # Tile (X, Y) shares a boundary with (X, Y+1) and (X-1, Y+1) or (X+1, Y+1).
    # Specifically, tile (X, Y) covers x-range [L, R].
    # Row Y+1 tiles cover [L', R'].
    # They are adjacent if [L, R] and [L', R'] overlap.
    # This means from (X, Y) you can reach (X, Y+1) and (X \pm 1, Y+1) in 1 step.
    # This is exactly the distance metric:
    # dist = max(|Y1 - Y2|, ceil(|X1 - X2| / 1)) ... no.
    # The distance is:
    # cost = max(|Y1 - Y2|, abs(X1 - X2) + (abs(Y1 - Y2)