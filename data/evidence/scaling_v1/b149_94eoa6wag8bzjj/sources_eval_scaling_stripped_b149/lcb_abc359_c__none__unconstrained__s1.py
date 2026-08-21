```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: A_{i,j} and A_{i+1,j} are in the same tile if i+j is even.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), (4,5)... if j is even
    # and pairs (1,2), (3,4), (5,6)... if j is odd.
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # For a cell (x, y), its tile ID can be represented as (X, Y).
    # If y is even, tiles are grouped as (x//2, y).
    # If y is odd, tiles are grouped as ((x-1)//2, y).
    # However, a simpler approach is to observe the distance.
    # Moving vertically always enters a new tile.
    # Moving horizontally might stay in the same tile.
    
    # Let's normalize the coordinates.
    # A cell (x, y) belongs to tile:
    # If (x + y) is even, it's the left half of a 2x1 tile.
    # If (x + y) is odd, it's the right half of a 2x1 tile.
    # The tile identity is:
    # If y is even: tile_x = x // 2, tile_y = y
    # If y is odd:  tile_x = (x + 1) // 2, tile_y = y
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # For j=0: (0,0)&(1,0), (2,0)&(3,0) ... -> tile_x = i // 2
    # For j=1: (1,1)&(2,1), (3,1)&(4,1) ... -> tile_x = (i+1) // 2
    # This is slightly wrong. Let's use the property:
    # A cell (x, y) is the "start" of a tile if (x+y) is even.
    # The tile containing (x, y) can be indexed by:
    # Y = y
    # X = x // 2 if y % 2 == 0 else (x + 1) // 2
    # But the rule says i+j is even. 
    # If j=0, i=0 is even -> A_{0,0} and A_{1,0} are one tile.
    # If j=1, i=1 is even -> A_{1,1} and A_{2,1} are one tile.
    # So for any j, the tile boundaries are at x = k*2 + (j % 2).
    
    # Let's redefine:
    # For a cell (x, y), its tile coordinates (X, Y) are:
    # Y = y
    # X = (x - (y % 2)) // 2
    # However, we must handle the offset carefully.
    # If y is even, boundaries are ..., -2, 0, 2, ...
    # If y is odd, boundaries are ..., -1, 1, 3, ...
    
    # Let's use a simpler logic:
    # The cost to move from (sx, sy) to (tx, ty) is:
    # cost = abs(sy - ty) + (1 if the tiles at the final y-level are different and 
    #                        we didn't already "pass through" that x-range during y-travel)
    # Actually, the most robust way is to map (x, y) to a coordinate (X, Y) 
    # such that moving from (X, Y) to (X', Y') costs |X-X'| + |Y-Y'|.
    # But the tiles are 2x1, so moving X costs 1 per 2 units.
    
    # Correct Tile Mapping:
    # For cell (x, y):
    # If y is even, it's in tile X = x // 2, Y = y
    # If y is odd, it's in tile X = (x + 1) // 2, Y = y
    # Wait, if y is odd, i+j even means i+1 is even, so i is odd.
    # A_{1,1} and A_{2,1} are one tile.
    # For y=1: x=0 is alone, x=1&2 are together, x=3&4 are together.
    # So for y=1, tile X is: 0 for x=0, 1 for x=1,2, 2 for x=3,4...
    # This is X = (x + 1) // 2.
    
    # Let's check:
    # y=0: x=0,1 -> X=0; x=2,3 -> X=1. Formula: x // 2.
    # y=1: x=0 -> X=0; x=1,2 -> X=1; x=3,4 -> X=2. Formula: (x + 1) // 2.
    
    # Now we have start (Xs, Ys) and end (Xt, Yt).
    # The distance is |Ys - Yt| + |Xs - Xt|.
    # But we must check if we are already in the same tile.
    # The cost is the number of NEW tiles entered.
    # If we start in tile (Xs, Ys), that tile is free.
    # Every move to a different tile costs 1.
    # The distance between (Xs, Ys) and (Xt, Yt)$ in a grid is $|Xs-Xt| + |Ys-Yt|$.
    
    # Let's refine Xs, Xt:
    # Xs = sx // 2 if sy % 2 == 0 else (sx + 1) // 2
    # Xt = tx // 2 if ty % 2 == 0 else (tx + 1) // 2
    # Ys = sy
    # Yt = ty
    
    # The distance is abs(Xs - Xt) + abs(Ys - Yt).
    # However, there is a special case: if we move vertically and the 
    # tile we land in is the same as the one we started in (because of the 2x1 shape),
    # but the problem says tiles are 2x1, so they only span two X-cells.
    # They never span two Y-cells. So every vertical move always enters a new tile.
    
    # Final check on the logic:
    # Sample 1: (5,0) to (2,5)
    # Ys=0, Xs=5//2 = 2.
    # Yt=5, Xt=(2+1)//2 = 1.
    # Dist = |2-1| + |0-5| = 1 + 5 = 6. 
    # Wait, Sample 1 output is 5. Why?
    # Because the starting tile is free.
    # The number of tiles entered is the distance.
    # If we are already in the target tile, cost is 0.
    # Otherwise, the cost is the Manhattan distance between tile coordinates.
    # Let's re-evaluate Sample 1:
    # Start: (5,0). Y=0 (even), X = 5//2 = 2. Tile (2, 0).
    # End: (2,5). Y=5 (odd), X = (2+1)//2 = 1. Tile (1, 5).
    # Distance = |2-1| + |0-5| = 6.
    # Still 6. Let's re-read. "Each time he enters a tile, he pays a toll of 1."
    # This means the starting tile is not paid for.
    # But the distance formula counts the number of edges.
    # In a grid, the number of tiles visited is distance + 1.
    # The number of tiles ENTERED is exactly the distance.
    # So why is Sample 1 output 5?
    # Let's trace: (5,0) -> (4,0) [Same tile (2,0)] -> (4,1) [Tile (2,1)] -> (3,1) [Same tile (2,1)]
    # -> (3,2) [Tile (1,2)] -> (3,3) [Tile (2,3)] -> (3,4) [Tile (2,4)] -> (3,5) [Tile (2,5)] -> (2,5) [Same tile (1,5)]
    # This is confusing. Let's use the coordinate system:
    # Tile(x, y) = (x // 2 if y % 2 == 0 else (x + 1) // 2, y)
    # Sample 1: S(5,0) -> Tile(2, 0). T(2,5) -> Tile(1, 5).
    # The distance is |2-1| + |0-5| = 6.
    # Is there a shorter path?
    # From (2,0), we can go to (2,1) [cost 1], then (1,1) [cost 0], then (1,2) [cost 1]...
    # Wait, if we are at (2,1), the tile is X=(2+1)//2 = 1.
    # So (5,0) [Tile 2,0] -> (4,0) [Tile 2,0] -> (4,1) [Tile 2,1] -> (3,1) [Tile 2,1] -> (3,2) [Tile 1,2]...
    # Let's use the property: we can move to any cell in the current tile for free.
    # From Tile(X, Y), we can move to:
    # 1. Tile(X, Y+1) - cost 1
    # 2. Tile(X, Y-1) - cost 1
    # 3. Tile(X+1, Y) - cost 1
    # 4. Tile(X-1, Y) - cost 1
    # BUT, we can also move to Tile(X, Y+1) or Tile(X, Y-1) and potentially 
    # land in a tile that allows a "free" horizontal move.
    # Actually, the only way to get 5 is if the distance is 5.
    # Let's re-calculate Tile(2,5): Y=5 (odd), X=(2+1)//2 = 1.
    # Let's re-calculate Tile(5,0): Y=0 (even), X=5//2 = 2.
    # Distance = |2-1| + |0-5| = 6.
    # What if we move to Y=1 first?
    # (5,0) [Tile 2,0] -> (5,1) [Tile (5+1)//2, 1] = [Tile 3,1].
    # Then from Tile(3,1) to Tile(1,5) is |3-1| + |1-5| = 2 + 4 = 6.
    # What if we move to (4,0) [Tile 2,0] -> (4,1) [Tile (4+1)//2, 1] = [Tile 2,1]?
    # Then from Tile(2,1) to Tile(1,5) is |2-1| + |1-5| = 1 + 4 = 5.
    # Total cost = 1 (to enter Tile(2,1)) + 5 = 6. Still 6.
    # Wait! The move (4,0) -> (4,1) enters Tile(2,1).
    # Then from Tile(2,1), we can move to Tile(1,2) by going to (3,1) -> (3,2).
    # Let's use the logic: the cost is min(dist(Tile(x,y), Tile(tx,ty))) 
    # over all (x,y) in the starting tile.
    # Starting tile is {(5,0), (4,0)}.
    # Tile(5,0) = (2,0). Tile(4,0) = (2,0).
    # Target tile is Tile(2,5) = (1,5).
    # Distance is |2-1| + |0-5| = 6.
    # There must be a mistake in my Tile calculation.
    # "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # j=0: i=0, 2, 4... are