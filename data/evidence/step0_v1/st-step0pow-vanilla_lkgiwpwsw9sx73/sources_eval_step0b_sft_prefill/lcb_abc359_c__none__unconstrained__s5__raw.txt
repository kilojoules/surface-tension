```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the x-boundaries of tiles are at:
    # j even: 0, 2, 4... (tiles are [0,2], [2,4]...)
    # j odd:  1, 3, 5... (tiles are [-1,1], [1,3]...)
    # Actually, the rule is: A_{i,j} and A_{i+1,j} are same if i+j is even.
    # Let's define the tile index for a cell (i, j):
    # The x-coordinate of the tile is (i + (i+j)%2) // 2
    # The y-coordinate of the tile is j
    # Wait, the problem says "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # This means for a fixed j, the pairs are (0,1), (2,3)... if j is even, and (-1,0), (1,2)... if j is odd.
    # Let f(i, j) be the index of the tile containing A_{i,j}.
    # f(i, j) = ((i + (j % 2)) // 2, j)
    
    # Let S = (sx, sy) and T = (tx, ty).
    # We want to find the minimum number of tiles entered.
    # Moving within a tile is free.
    # Moving from tile (ux, uy) to (vx, vy) costs 1 if (ux, uy) != (vx, vy).
    # However, we can move any distance n in one direction.
    # If we move vertically from (ux, uy) to (ux, vy), we enter |uy - vy| tiles.
    # If we move horizontally from (ux, uy) to (vx, uy), we enter |ux - vx| tiles.
    
    # Let's transform the coordinates to the "tile" coordinate system.
    # A cell (i, j) belongs to tile X = (i + (j % 2)) // 2, Y = j.
    # To move from (sx, sy) to (tx, ty):
    # We can move to some (sx, ty) then to (tx, ty), or (tx, sy) then (tx, ty).
    # But we can zigzag.
    
    # The distance is effectively the L1 distance in the tile grid, 
    # but we must be careful about the offset.
    # Let g(i, j) = (i + (j % 2)) // 2.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # min( |g(sx, sy) - g(tx, sy)| + |sy - ty|, 
    #      |g(sx, ty) - g(tx, ty)| + |sy - ty| )
    # Wait, that's not quite right. If we move vertically first, we change the "parity" 
    # of the row, which changes the x-boundary.
    
    # Let's use the property:
    # To get from (sx, sy) to (tx, ty), we must cover the vertical distance |sy - ty|.
    # Each vertical step enters a new tile.
    # The horizontal distance is covered by the difference in tile indices.
    # Let X1 = (sx + (sy % 2)) // 2 and Y1 = sy
    # Let X2 = (tx + (ty % 2)) // 2 and Y2 = ty
    
    # The minimum cost is |Y1 - Y2| + max(0, |X1 - X2| - 1) if we can align the 
    # horizontal move to a row that "bridges" the two tiles.
    # Actually, a simpler way to think about it:
    # You must pay for every change in Y. That's |sy - ty|.
    # Additionally, you might need to pay for the horizontal distance.
    # If you are at (sx, sy) and move to (tx, sy), the cost is |g(sx, sy) - g(tx, sy)|.
    # Then moving from (tx, sy) to (tx, ty) costs |sy - ty|.
    # Total = |(sx + (sy%2))//2 - (tx + (sy%2))//2| + |sy - ty|
    
    # But you can pick ANY row k between sy and ty to do the horizontal transition.
    # Cost = |sy - k| + |(sx + (k%2))//2 - (tx + (k%2))//2| + |k - ty|
    # Since |sy - k| + |k - ty| = |sy - ty| for k between sy and ty,
    # we want to minimize |(sx + (k%2))//2 - (tx + (k%2))//2| for k in {sy, ty}.
    # (Because k%2 only takes two values).
    
    # Let h(k) = |(sx + (k % 2)) // 2 - (tx + (k % 2)) // 2|
    # Min Cost = |sy - ty| + min(h(sy), h(ty))
    # Note: if sy == ty, the cost is just h(sy). But if sy != ty, 
    # we can transition at either the start row or the end row.
    # Actually, if sy != ty, we can transition at any k. But k%2 only oscillates.
    # If |sy - ty| >= 1, we can pick k such that k%2 is 0 or 1.
    # So we take min(h(0), h(1)) where h(p) = |(sx + p)//2 - (tx + p)//2|.
    
    # Let's refine:
    # If sy == ty: cost is |(sx + (sy%2))//2 - (tx + (sy%2))//2|
    # If sy != ty: cost is |sy - ty| + min(
    #    |(sx + (sy%2))//2 - (tx + (sy%2))//2|, 
    #    |(sx + (1-sy%2))//2 - (tx + (1-sy%2))//2|
    # )
    # Wait, the second term in the min is the cost if we move 1 unit vertically first.
    # But moving 1 unit vertically costs 1.
    # Let's use the property:
    # Cost = |sy - ty| + max(0, |(sx + (sy%2))//2 - (tx + (ty%2))//2| - 0) ... no.
    
    # Correct logic:
    # To go from (sx, sy) to (tx, ty), you must traverse |sy - ty| boundaries.
    # This puts you in tile (g(sx, sy), sy) and eventually (g(tx, ty), ty).
    # The number of horizontal tiles crossed is |g(sx, sy) - g(tx, ty)|.
    # However, one of these might be "free" if the vertical movement aligns.
    # The distance is |sy - ty| + |(sx + (sy%2))//2 - (tx + (ty%2))//2|.
    # But we can potentially reduce this by 1 if we can move horizontally in a row 
    # that "overlaps" both the start and end x-ranges.
    
    # Let's use the provided Sample 1: 5 0 -> 2 5
    # sx=5, sy=0, tx=2, ty=5
    # g(5, 0) = (5+0)//2 = 2. g(2, 5) = (2+1)//2 = 1.
    # |0 - 5| + |2 - 1| = 5 + 1 = 6.
    # Sample 1 output is 5. Why?
    # Because at sy=0, tile is [4, 6). At sy=1, tile is [3, 5).
    # If we move (5, 0) -> (4, 0) [cost 0], then (4, 0) -> (4, 1) [cost 1],
    # then (4, 1) -> (2, 1) [cost 0, since (4,1) and (3,1) and (2,1) are not same], 
    # wait.
    # Let's re-evaluate:
    # Row 0: tiles are [0,2), [2,4), [4,6)...
    # Row 1: tiles are [-1,1), [1,3), [3,5), [5,7)...
    # (5, 0) is in tile (2, 0). (2, 5) is in tile (1, 5).
    # Path: (5, 0) -> (4, 0) [Tile (2,0)] -> (4, 1) [Tile (2,1)] -> (2, 1) [Tile (1,1)] -> (2, 5) [Tile (1,5)]
    # Costs: Enter (2,1) cost 1, Enter (1,1) cost 1, Enter (1,2) cost 1, Enter (1,3) cost 1, Enter (1,4) cost 1, Enter (1,5) cost 1.
    # Total 6? No.
    # Sample 1: (5, 0) -> (4, 0) [0] -> (4, 1) [1] -> (3, 1) [0] -> (3, 4) [3] -> (2, 4) [0] -> (2, 5) [1]. Total 5.
    # The formula is: cost = |sy - ty| + |(sx + (sy%2))//2 - (tx + (ty%2))//2|
    # For Sample 1: |0 - 5| + |(5+0)//2 - (2+1)//2| = 5 + |2 - 1| = 6. Still 6.
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # i=0, j=0: A_{0,0}, A_{1,0} same. i=2, j=0: A_{2,0}, A_{3,0} same.
    # i=0, j=1: A_{0,1} is alone? No. i+j even => 0+1 is odd.
    # i=1, j=1: A_{1,1}, A_{2,1} same.
    # So for j=0, tiles are {0,1}, {2,3}, {4,5}...
    # For j=1, tiles are {1,2}, {3,4}, {5,6}...
    # (5, 0) is in tile {4,5}. (2, 5) is in tile {2,3}.
    # Let's use the coordinate transformation:
    # A cell (i, j) is in tile ( (i + (j%2)) // 2, j )
    # Sample 1: S=(5,0) -> tile ( (5+0)//2, 0 ) = (2, 0). T=(2,5) -> tile ( (2+1)//2, 5 ) = (1, 5).
    # The distance between (x1, y1) and (x2, y2) in this grid is |x1-x2| + |y1-y2|.
    # But we can move diagonally? No.
    # We can move to any (x, y) in the same tile for free.
    # From tile (x, y), we can move to (x, y+1), (x, y-1), or (x+1, y), (x-1, y).
    # Wait, from tile (x, y), we can reach any cell (i, j) in it.
    # From such a cell, we can move to (i, j+1) which is in tile ( (i+(j+1)%2)//2, j+1 ).
    # This means from (x, y), we can reach tiles:
    # 1. (x, y+1) if (i + (j+1)%2)//2 == x for some i in tile (x, y)
    # 2. (x, y-1) if (i + (j-1)%2