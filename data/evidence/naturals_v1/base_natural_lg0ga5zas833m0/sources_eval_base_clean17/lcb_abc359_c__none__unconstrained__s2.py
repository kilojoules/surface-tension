```python
import sys

def solve():
    """
    The tiling pattern:
    A_{i,j} is the square [i, i+1] x [j, j+1].
    If i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    This means for a fixed row j:
    If j is even: (0,0)-(1,0), (2,0)-(3,0), (4,0)-(5,0)... are tiles.
    If j is odd: (1,1)-(2,1), (3,1)-(4,1), (5,1)-(6,1)... are tiles.
    
    Let's define a coordinate system for the tiles.
    A tile T can be identified by its "base" cell (i, j) where the 2x1 tile starts.
    For a cell (x, y), the tile it belongs to is:
    If (x+y) is even, the tile is {(x, y), (x+1, y)}. Let's represent this tile by (x, y).
    If (x+y) is odd, the tile is {(x-1, y), (x, y)}. Let's represent this tile by (x-1, y).
    
    Wait, a simpler way:
    A cell (x, y) belongs to tile (i, j) where:
    i = x if (x+y) is even else x-1
    j = y
    
    Let's check:
    If x+y is even: i = x, j = y. Tile is {(x, y), (x+1, y)}.
    If x+y is odd: i = x-1, j = y. Tile is {(x-1, y), (x, y)}.
    In both cases, the tile is a 2x1 horizontal block.
    
    Now, moving:
    - Move Right/Left: 
      If we are in tile (i, j) and move to cell (x', y), we check if (x', y) is in the same tile.
      Moving horizontally within a tile costs 0. Moving to a new tile costs 1.
    - Move Up/Down:
      Moving to a new row j' always enters a new tile (since tiles are 2x1 horizontal).
    
    Actually, this is a shortest path problem on a grid of tiles.
    Let's map cell (x, y) to tile coordinates (U, V).
    Let U = x // 2 and V = y.
    If y is even:
        Cells (2U, y) and (2U+1, y) are in the same tile.
    If y is odd:
        Cells (2U-1, y) and (2U, y) are in the same tile.
    
    This is slightly confusing. Let's use the property:
    Toll is paid when entering a NEW tile.
    Starting cell (Sx, Sy) is already in a tile.
    Target cell (Tx, Ty) is in a tile.
    
    Let's transform (x, y) to a coordinate system where tiles are 1x1.
    For a cell (x, y), let its tile be (X, Y).
    If y is even: X = x // 2, Y = y
    If y is odd: X = (x - 1) // 2, Y = y
    Wait, if y is odd and x=0, (x-1)//2 = -1. This works.
    
    Now, what are the costs to move between tiles (X, Y)?
    From (X, Y) to (X+1, Y):
    - If Y is even: Cell (2X+1, Y) and (2X+2, Y). 
      (2X+1, Y) is in tile (X, Y). (2X+2, Y) is in tile (X+1, Y).
      Moving from (2X+1, Y) to (2X+2, Y) costs 1.
    - If Y is odd: Cell (2X, Y) and (2X+1, Y).
      (2X, Y) is in tile (X, Y). (2X+1, Y) is in tile (X+1, Y).
      Moving from (2X, Y) to (2X+1, Y) costs 1.
    So horizontal distance is |X1 - X2|.
    
    From (X, Y) to (X, Y+1):
    - Cell (x, Y) and (x, Y+1).
      Are they in the same tile? No, tiles are 2x1 horizontal.
      Does moving from (x, Y) to (x, Y+1) always cost 1?
      Let's see.
      Cell (x, Y) is in tile (X, Y).
      Cell (x, Y+1) is in tile (X', Y+1).
      If Y is even: X = x // 2.
      If Y+1 is odd: X' = (x-1) // 2.
      If x is even: X = x//2, X' = (x-1)//2 = X-1.
      If x is odd: X = x//2, X' = (x-1)//2 = X.
      So moving from (x, Y) to (x, Y+1) might land you in the same X or X-1.
      
    This is a distance problem on a graph.
    The distance is the number of tiles entered.
    The cost to move from (X, Y) to (X', Y') is:
    Dist = |Y - Y'| + max(0, |X - X'| - (1 if (Y+Y' is odd) else 0)) ? No.
    
    Let's reconsider:
    To move from (Sx, Sy) to (Tx, Ty):
    Vertical distance is |Sy - Ty|. Each vertical step enters a new tile.
    Horizontal distance:
    In each row, tiles are 2 units wide.
    The number of tiles to cross horizontally is roughly |Sx - Tx| / 2.
    
    Correct approach:
    The cost is the distance in a grid where some edges are 0 and some are 1.
    Actually, the problem can be simplified:
    The cost is |Sy - Ty| + distance_horizontal.
    The horizontal movement is "free" if we can align the tiles.
    
    Let's use the coordinate transformation:
    X = (x + (y % 2)) // 2
    Y = y
    Now we are moving from (X1, Y1) to (X2, Y2).
    Moving from (X, Y) to (X +/- 1, Y) costs 1.
    Moving from (X, Y) to (X, Y +/- 1) costs 1.
    Wait, if we move from (X, Y) to (X, Y+1), we are entering a new tile.
    But can we move from (X, Y) to (X, Y+1) and then to (X+1, Y+1) for the same cost?
    
    Let's use the logic from similar problems:
    The minimum cost is |Sy - Ty| + max(0, |X1 - X2| - 1) if we can leverage the offset.
    Actually, the distance is simply:
    dx = abs(X1 - X2)
    dy = abs(Y1 - Y2)
    Result = dy + max(0, dx - 1) if dy == 0 else dy + dx - 1? No.
    
    Let's use the manhattan distance on the transformed grid:
    Cost = |X1 - X2| + |Y1 - Y2|.
    But we can move diagonally? No.
    
    Let's re-evaluate:
    The cost to move from (X, Y) to (X, Y+1) is 1.
    The cost to move from (X, Y) to (X+1, Y) is 1.
    However, if we are at (X, Y) and want to go to (X+1, Y+1), we can:
    (X, Y) -> (X, Y+1) -> (X+1, Y+1) : cost 1 + 1 = 2.
    Or (X, Y) -> (X+1, Y) -> (X+1, Y+1) : cost 1 + 1 = 2.
    
    Wait, the sample 1: (5, 0) to (2, 5).
    S: x=5, y=0. X1 = (5 + 0)//2 = 2, Y1 = 0.
    T: x=2, y=5. X2 = (2 + 1)//2 = 1, Y2 = 5.
    |X1 - X2| = 1, |Y1 - Y2| = 5.
    Sample output: 5.
    If the formula was |X1-X2| + |Y1-Y2|, it would be 6.
    But here it's 5. This means one of the moves was "free".
    
    If we move vertically, we enter a new tile.
    But if we are at (x, y) in tile (X, Y) and move to (x, y+1), we enter tile (X', Y+1).
    If X' == X, we paid 1 toll.
    If X' != X, we still paid 1 toll.
    However, we can choose x such that X' == X.
    If y is even, X = x // 2.
    If y+1 is odd, X' = (x-1) // 2.
    For X = X', we need x // 2 = (x-1) // 2, which means x must be odd.
    If x is even, X = x // 2 and X' = x // 2 - 1.
    
    So, from (X, Y), we can reach (X, Y+1) or (X-1, Y+1) with cost 1.
    This means we can change X by 1 for free while changing Y by 1.
    This is like the Chebyshev distance if we can move diagonally.
    The cost is max(|X1 - X2|, |Y1 - Y2|).
    Let's check Sample 1: max(|2-1|, |0-5|) = max(1, 5) = 5. Correct!
    Sample 2: (3, 1) to (4, 1).
    S: x=3, y=1. X1 = (3 + 1)//2 = 2, Y1 = 1.
    T: x=4, y=1. X2 = (4 + 1)//2 = 2, Y2 = 1.
    max(|2-2|, |1-1|) = 0. Correct!
    Sample 3:
    S: 2552608206527595, 5411232866732612
    T: 771856005518028, 7206210729152763
    X1 = (2552608206527595 + 0)//2 = 1276304103263797
    Y1 = 5411232866732612
    X2 = (771856005518028 + 1)//2 = 385928002759014
    Y2 = 7206210729152763
    dx = |1276304103263797 - 385928002759014| = 890376100504783
    dy = |5411232866732612 - 7206210729152763| = 1794977862420151
    max(dx, dy) = 1794977862420151. Correct!
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    sx, sy = int(input_data[0]), int(input_data[1])
    tx, ty = int(input_data[2]), int(input_data[3])
    
    x1 = (sx + (sy % 2)) // 2
    y1 = sy
    x2 = (tx + (ty % 2)) // 2
    y2 = ty
    
    print(max(abs(x1 - x2), abs