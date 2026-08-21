```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, the tiles are {(0,j), (1,j)}, {(2,j), (3,j)} if j is even.
    # In row j, the tiles are {(-1,j), (0,j)}, {(1,j), (2,j)} if j is odd.
    
    # Let's normalize the coordinates so that in every row, 
    # tiles are {(2k, j), (2k+1, j)}.
    # If j is even, the offset is 0. If j is odd, the offset is -1.
    # A simpler way: a square (i, j) belongs to tile index (i + (j % 2)) // 2 in row j.
    
    # The cost to move between (sx, sy) and (tx, ty):
    # 1. Vertical distance: Each change in y always enters a new tile.
    #    Cost = |ty - sy|
    # 2. Horizontal distance:
    #    Let x_coord(i, j) = (i + (j % 2)) // 2
    #    The horizontal cost is the distance between x_coord(sx, sy) and x_coord(tx, ty).
    #    However, we can move vertically first, then horizontally, or vice versa.
    #    The total cost is |ty - sy| + max(0, dist_horizontal - 1) is NOT correct because
    #    we can optimize the horizontal transition by picking the best row to move in.
    
    # Correct logic:
    # The cost is |ty - sy| + the cost to move from the tile at (sx, sy) to the tile at (tx, ty).
    # Let U = (sx + (sy % 2)) // 2
    # Let V = (tx + (ty % 2)) // 2
    # The horizontal distance in terms of tiles is |U - V|.
    # If we are already in the target tile's column (U == V), cost is 0.
    # If we move to a different tile column, it costs |U - V|.
    # But wait, if we move vertically, we might enter the target column "for free" 
    # as part of the vertical cost.
    
    # Let's refine:
    # Total Cost = |ty - sy| + max(0, |U - V| - (1 if sy != ty else 0))
    # Actually, the simplest derivation for this specific tiling is:
    # Cost = |ty - sy| + max(0, abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2) - (1 if sy != ty else 0))
    # Wait, the "1 if sy != ty" is only if the vertical movement allows us to 
    # land in the target tile column.
    
    # Let's use the property:
    # Cost = |ty - sy| + max(0, abs(U - V) - (1 if sy != ty else 0))
    # Let's test Sample 1: 5 0 -> 2 5
    # U = (5 + 0)//2 = 2; V = (2 + (5%2))//2 = (2+1)//2 = 1
    # Cost = |5 - 0| + max(0, |2 - 1| - 1) = 5 + 0 = 5. Correct.
    
    # Sample 2: 3 1 -> 4 1
    # U = (3 + 1)//2 = 2; V = (4 + 1)//2 = 2
    # Cost = |1 - 1| + max(0, |2 - 2| - 0) = 0. Correct.
    
    # Sample 3: 2552608206527595 5411232866732612 -> 771856005518028 7206210729152763
    # sy = 5411232866732612, ty = 7206210729152763
    # sx = 2552608206527595, tx = 771856005518028
    # U = (2552608206527595 + 0) // 2 = 1276304103263797
    # V = (771856005518028 + 1) // 2 = 385928002759014
    # dy = 7206210729152763 - 5411232866732612 = 1794977862420151
    # dist_u_v = 1276304103263797 - 385928002759014 = 890376100504783
    # Cost = 1794977862420151 + max(0, 890376100504783 - 1) = 1794977862420151 + 890376100504782
    # This doesn't match Sample 3 (1794977862420151). 
    # Re-evaluating: The vertical movement itself enters new tiles.
    # If we move from (sx, sy) to (tx, ty), we must cross |ty - sy| boundaries.
    # Each vertical step enters a new tile. 
    # The horizontal distance is |U - V|. 
    # If we move vertically, we can change our "tile column" U.
    # In row j, the tile column is (x + (j%2)) // 2.
    # This means in row j, x is in tile U. In row j+1, x is in tile (x + (j+1)%2) // 2.
    # This is either U or U+1.
    # So vertical movement allows us to shift our tile-column by 1 for free.
    # The minimum cost is max(|ty - sy|, |ty - sy| + |U - V| - 1) if sy != ty else |U - V|
    # Wait, the correct formula for this problem is:
    # Cost = max(|ty - sy|, |ty - sy| + |U - V| - 1) if sy != ty else |U - V|
    # Let's check Sample 3 again:
    # dy = 1794977862420151, |U-V| = 890376100504783
    # Cost = max(1794977862420151, 1794977862420151 + 890376100504783 - 1)
    # Still too high. Let's re-read. "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # If he moves from (sx, sy) to (tx, ty):
    # He can move to a row j, then move horizontally to the target column, then move to ty.
    # The cost is |sy - j| + |tx_tile(j) - sx_tile(j)| + |ty - j|.
    # To minimize this, j should be between sy and ty.
    # Cost = |ty - sy| + |tx_tile(j) - sx_tile(j)|.
    # We want to minimize |(tx + (j%2))//2 - (sx + (j%2))//2|.
    # Let f(j) = (tx + (j%2))//2 - (sx + (j%2))//2.
    # If j is even: f(0) = tx//2 - sx//2.
    # If j is odd: f(1) = (tx+1)//2 - (sx+1)//2.
    # The cost is |ty - sy| + min(|f(0)|, |f(1)|) if sy != ty else |f(sy%2)|.
    # Wait, if sy == ty, he is already in row sy, so he must use f(sy%2).
    # If sy != ty, he can pick either j=sy or j=ty or any j in between.
    # Actually, he can pick any j such that he minimizes |f(j)|.
    # Since j%2 only takes values 0 and 1, he can always pick the better of f(0) and f(1) as long as he moves at least one unit vertically.
    # But he can only pick j%2 = 0 if there is an even j between sy and ty, and j%2 = 1 if there is an odd j.
    # If |ty - sy| >= 1, both 0 and 1 are available as j%2.
    
    # Let's re-test Sample 3:
    # sx = 2552608206527595, sy = 5411232866732612
    # tx = 771856005518028, ty = 7206210729152763
    # f(0) = 771856005518028//2 - 2552608206527595//2 = 385928002759014 - 1276304103263797 = -890376100504783
    # f(1) = (771856005518028+1)//2 - (2552608206527595+1)//2 = 385928002759014 - 1276304103263798 = -890376100504784
    # min(|f(0)|, |f(1)|) = 890376100504783
    # Total = 1794977862420151 + 890376100504783... still not 1794977862420151.
    
    # Let's re-read again. "Each time he enters a tile, he pays a toll of 1."
    # If he is in tile T1 and moves to T2, he pays 1.
    # If he moves and stays in T1, he pays 0.
    # This is a shortest path problem on a graph where nodes are tiles.
    # Tile (i, j) is defined by: if (i+j) is even, tile is {(i,j), (i+1,j)}.
    # Let's use a coordinate transformation:
    # A square (i, j) belongs to tile (X, Y) where:
    # Y = j
    # X = (i + (j % 2)) // 2
    # Neighbors of tile (X, Y):
    # 1. Horizontal: (X-1, Y) and (X+1, Y). Cost = 1.
    # 2. Vertical: 
    #    Square (i, j) is in tile (X, Y).
    #    Square (i, j+1) is in tile (X', Y+1) where X' = (i + ((j+1)%2)) // 2.
    #    Square (i, j-1) is in tile (X'', Y-1) where X'' = (i + ((j-1)%2)) // 2.
    #    This means tile (X, Y) is connected to tiles in row Y+1 and