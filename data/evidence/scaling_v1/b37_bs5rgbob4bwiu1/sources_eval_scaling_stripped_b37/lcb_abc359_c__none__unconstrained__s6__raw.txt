```python
import sys

def solve():
    # Read input and parse integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1 rectangles. 
    # A square (i, j) and (i+1, j) are in the same tile if i+j is even.
    # This means for a fixed j:
    # If j is even, tiles are {(0,j), (1,j)}, {(2,j), (3,j)}, ...
    # If j is odd, tiles are {(-1,j), (0,j)}, {(1,j), (2,j)}, ...
    
    # We can represent the state by (tile_x, tile_y).
    # For a given (x, y), the tile index in the x-direction depends on the parity of y.
    # If y is even, the tile index is x // 2.
    # If y is odd, the tile index is (x + 1) // 2.
    # However, a simpler way to think about it is:
    # A move in x costs 0 if you stay within the same 2x1 tile.
    # A move in y always enters a new tile unless you are already in one.
    # The distance is essentially the Manhattan distance in a transformed coordinate system.
    
    # Let's define a coordinate transformation to a grid where each cell is a 2x1 tile.
    # For a point (x, y), the tile coordinates (u, v) are:
    # v = y
    # u = (x + (y % 2)) // 2
    
    # The cost to move from (sx, sy) to (tx, ty):
    # 1. Vertical distance: |sy - ty|
    # 2. Horizontal distance: 
    #    The number of tile boundaries crossed horizontally.
    #    In one row y, the boundaries are at x = k*2 if y is even, and x = k*2 - 1 if y is odd.
    #    The number of boundaries between sx and tx in row y is |u_s - u_t|.
    #    Wait, the problem says "Each time he enters a tile, he pays a toll of 1."
    #    Starting tile is free.
    #    Moving from (sx, sy) to (tx, ty):
    #    Total cost = |sy - ty| + (distance in u)
    #    But we can optimize: we can move to a row where the horizontal distance is minimized.
    #    Actually, the cost is simply:
    #    abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    #    Wait, that's not quite right because we can change rows.
    #    The correct logic for this specific tiling pattern:
    #    The distance is max(|sy - ty|, abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2) * 2 - (something))
    #    Actually, the simplest formula for this problem is:
    #    cost = abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    #    Let's check Sample 1: 5 0 -> 2 5
    #    sy=0, sx=5 => u_s = (5 + 0)//2 = 2
    #    ty=5, tx=2 => u_t = (2 + 1)//2 = 1
    #    cost = |0 - 5| + |2 - 1| = 5 + 1 = 6. 
    #    Sample 1 output is 5. Why?
    #    Because we can move to a row where the u-coordinates are closer.
    #    If we are at (5, 0), u=2. If we move to (5, 1), u=(5+1)//2 = 3.
    #    The distance is actually:
    #    dist = abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    #    But we can choose to move to sy+1 or sy-1 first.
    #    The parity of the row affects the u-coordinate.
    #    Let f(x, y) = (x + (y % 2)) // 2
    #    Cost = min(
    #        abs(sy - ty) + abs(f(sx, sy) - f(tx, ty)),
    #        abs(sy + 1 - ty) + abs(f(sx, sy + 1) - f(tx, ty)),
    #        abs(sy - 1 - ty) + abs(f(sx, sy - 1) - f(tx, ty))
    #    )
    #    Wait, the most general form is:
    #    The cost is abs(sy - ty) + abs(f(sx, sy) - f(tx, ty)) 
    #    BUT, if we change the parity of y, the f(x, y) changes.
    #    The minimum cost is actually:
    #    abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    #    Wait, let's re-evaluate Sample 1: (5,0) to (2,5).
    #    f(5, 0) = 2. f(2, 5) = (2+1)//2 = 1.
    #    Cost = |0-5| + |2-1| = 6.
    #    If we move to (5, 1) first: cost 1. Then f(5, 1) = 3.
    #    Then to (2, 5): |1-5| + |3-1| = 4 + 2 = 6. Total 7.
    #    If we move to (4, 0) first: cost 0 (same tile). f(4, 0) = 2.
    #    Wait, (5,0) and (6,0) are in the same tile. (4,0) and (5,0) are NOT.
    #    A_{i,j} and A_{i+1,j} are same if i+j is even.
    #    For j=0: (0,0)&(1,0), (2,0)&(3,0), (4,0)&(5,0).
    #    So (5,0) is in tile index 2 (0-indexed).
    #    For j=5: (1,5)&(2,5), (3,5)&(4,5).
    #    So (2,5) is in tile index 0 (since 1+5=6 is even, A_{1,5} and A_{2,5} are together).
    #    Wait, if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    #    For j=5, i=1 is even? No, 1+5=6 is even. So A_{1,5} and A_{2,5} are one tile.
    #    The tile index for (x,y) is:
    #    If (x+y) is even, it's the left tile of the pair: index (x - (x+y)%2) // 2
    #    Actually: the pair is {i, i+1} where i+j is even.
    #    So i = x if x+j is even, else i = x-1.
    #    i = x - ((x + j) % 2)
    #    Tile index u = i // 2.
    #    Let's use: u = (x - (x + y) % 2) // 2
    #    Sample 1: S(5,0), T(2,5)
    #    u_s = (5 - (5+0)%2) // 2 = (5-1)//2 = 2
    #    u_t = (2 - (2+5)%2) // 2 = (2-1)//2 = 0
    #    Cost = |0-5| + |2-0| = 5 + 2 = 7. Still not 5.
    #    Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    #    Starting tile is free.
    #    Moving from (sx, sy) to (tx, ty):
    #    The number of tiles entered is the number of boundaries crossed.
    #    Vertical boundaries are at x = i where i+j is odd.
    #    Horizontal boundaries are always at y = j.
    #    This is a shortest path problem on a graph.
    #    The distance is simply:
    #    abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    #    Wait, the sample 1 explanation says:
    #    (5,0) -> (4,0) [Left 1]: (4,0) and (5,0) are in the same tile because 4+0=4 (even).
    #    Toll: 0.
    #    (4,0) -> (4,1) [Up 1]: New tile. Toll: 1.
    #    (4,1) -> (3,1) [Left 1]: (3,1) and (4,1) are in the same tile because 3+1=4 (even).
    #    Toll: 0.
    #    (3,1) -> (3,4) [Up 3]: 3 tiles. Toll: 3.
    #    (3,4) -> (2,4) [Left 1]: (2,4) and (3,4) are in the same tile because 2+4=6 (even).
    #    Toll: 0.
    #    (2,4) -> (2,5) [Up 1]: New tile. Toll: 1.
    #    Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # Let's use the coordinate transformation:
    # A tile is identified by (u, v) where v = y and u = (x + (y % 2)) // 2.
    # Moving from (u, v) to (u', v'):
    # If v == v', cost is |u - u'|.
    # If u == u', cost is |v - v'|.
    # But we can change u by changing v.
    # From (u, v), moving to (u, v+1) might change the tile index u.
    # New u' = (x + ((y+1) % 2)) // 2.
    # Since (x + (y%2)) // 2 = u, then x is either 2u or 2u-1 (depending on y%2).
    # If y%2 == 0, x is 2u or 2u+1. Then (x+1)//2 is u or u+1.
    # If y%2 == 1, x is 2u-1 or 2u. Then x//2 is u-1 or u.
    # This means moving v -> v+1 allows u -> u or u -> u +/- 1.
    # The minimum cost is simply:
    # abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    # Wait, I just calculated that as 6 for Sample 1. Let me re-calculate.
    # sx=5, sy=0 => u_s = (5 + 0)//2 = 2
    # tx=2, ty=5 => u_t = (2 + 1)//2 = 1
    # |0-5| + |2-1| = 5 + 1 = 6.
    # Still 6. Let me re-read the sample explanation again.
    # (5,0) is in tile A_{4,0} U A_{5,0}. So u=2, v=0.
    # (2,5) is in tile A_{1,5} U A_{2,5}. So u=1, v=5.
    # The sample says we can reach it in 5.
    # (5,0) [u=2, v=0] -> (4,0) [u=2, v=0] -> (4,1) [u=2, v=1] -> (3,1) [u=