```python
import sys

def solve():
    """
    The problem describes a grid of 2x1 tiles.
    Rule: When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    This means for a fixed j:
    - If j is even: tiles are {(0,j),(1,j)}, {(2,j),(3,j)}, ... (i is even)
    - If j is odd: tiles are {(-1,j),(0,j)}, {(1,j),(2,j)}, ... (i is odd)
    
    Essentially, for any cell (i, j), it belongs to a tile defined by:
    If (i+j) is even, it's the left half of a tile (or the only half if we consider boundaries).
    Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    Let's re-evaluate:
    Cell (i, j) is part of a 2x1 tile.
    If i+j is even, (i, j) and (i+1, j) are one tile.
    If i+j is odd, (i, j) and (i-1, j) are one tile.
    
    Let's define the tile ID for cell (i, j):
    If (i+j) % 2 == 0: Tile ID is ( (i // 2 if j%2==0 else (i-1)//2), j ) 
    Actually, simpler:
    A cell (i, j) belongs to tile T = (x_tile, y_tile) where:
    y_tile = j
    If j is even: x_tile = i // 2 (where i is the coordinate, so 0,1 -> 0; 2,3 -> 1)
    If j is odd: x_tile = (i - 1) // 2 (so -1,0 -> -1; 1,2 -> 0)
    
    Wait, let's check the rule: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    If j=0 (even), i=0 is even, so A_{0,0} and A_{1,0} are one tile.
    If j=1 (odd), i=0 is odd? No, i=0, j=1 => i+j=1 (odd).
    If j=1, i=1 is odd, so i+j=2 (even), so A_{1,1} and A_{2,1} are one tile.
    
    Correct logic:
    For a fixed j:
    If j is even, tiles are [0,1], [2,3], [4,5]...
    If j is odd, tiles are [-1,0], [1,2], [3,4]...
    
    Let's map cell (i, j) to a coordinate (X, Y) in a grid of tiles:
    Y = j
    If j % 2 == 0: X = i // 2
    If j % 2 != 0: X = (i + 1) // 2 if i % 2 == 0 else (i - 1) // 2
    Wait, simpler: if j is odd, the boundaries are at ...-1, 1, 3, 5...
    So for j odd, X = (i + 1) // 2 is not quite right.
    Let's use:
    If j % 2 == 0: X = i // 2
    If j % 2 != 0: X = (i + 1) // 2 if i % 2 == 0 else (i - 1) // 2
    Actually, if j is odd:
    i=0: i+j=1 (odd). i=1: i+j=2 (even). So (1,j) and (2,j) are together.
    i=0 is with i=-1.
    So for j odd:
    i=0 -> tile -1
    i=1 -> tile 0
    i=2 -> tile 0
    i=3 -> tile 1
    i=4 -> tile 1
    This means for j odd, X = (i + 1) // 2 if i is even? No.
    Let's use:
    If j % 2 == 0: X = i // 2
    If j % 2 != 0: X = (i + 1) // 2 if i % 2 == 0 else (i - 1) // 2
    Let's check:
    j=1, i=0: (0+1)//2 = 0? No, should be -1.
    Correct mapping for j odd:
    i=0: i+j=1 (odd) -> paired with i-1. X = (0-1)//2 = -1.
    i=1: i+j=2 (even) -> paired with i+1. X = (1-1)//2 = 0.
    i=2: i+j=3 (odd) -> paired with i-1. X = (2-1)//2 = 0.
    i=3: i+j=4 (even) -> paired with i+1. X = (3-1)//2 = 1.
    So for j odd: X = (i - 1) // 2 if i is odd else (i - 1) // 2.
    Wait: (1-1)//2 = 0, (2-1)//2 = 0. (3-1)//2 = 1, (4-1)//2 = 1.
    So for j odd: X = (i - 1) // 2.
    Let's re-verify:
    j=0: i=0,1 -> X=0; i=2,3 -> X=1. (X = i // 2)
    j=1: i=0 -> X=-1; i=1,2 -> X=0; i=3,4 -> X=1. (X = (i-1) // 2)
    
    Now we have start (Sx, Sy) -> (X1, Y1) and end (Tx, Ty) -> (X2, Y2).
    The distance is the min toll.
    Moving in Y direction: each step changes Y by 1, entering a new tile.
    Moving in X direction: each step changes X by 1, entering a new tile.
    However, moving diagonally (one X, one Y) might be cheaper.
    
    Wait, the movement is: "Choose a direction... and a positive integer n. Move n units."
    This means we can move any distance in one of the 4 cardinal directions.
    Entering a tile costs 1.
    If we are at (X1, Y1) and move to (X2, Y2):
    The cost to change Y by 1 is 1.
    The cost to change X by 1 is 1.
    But if we move from (X, Y) to (X+1, Y+1), we can go (X,Y) -> (X, Y+1) -> (X+1, Y+1) cost 2,
    or (X,Y) -> (X+1, Y) -> (X+1, Y+1) cost 2.
    Is there a way to move "diagonally" cheaper?
    The problem is equivalent to finding the distance in a grid where some edges are weighted.
    Actually, this is a known problem. The distance between (X1, Y1) and (X2, Y2) in this specific tile layout is:
    dist = max(|X1 - X2|, |Y1 - Y2|, (|X1 - X2| + |Y1 - Y2| + 1) // 2) - No, that's for different grids.
    
    Let's look at the cost:
    To move from (X, Y) to (X+1, Y), the cost is 1.
    To move from (X, Y) to (X, Y+1), the cost is 1.
    But can we move from (X, Y) to (X+1, Y+1) with cost 1?
    No, because we must move in a cardinal direction.
    To get to (X+1, Y+1), we must move Right then Up (or Up then Right).
    Right: (X,Y) -> (X+1, Y) costs 1.
    Up: (X+1, Y) -> (X+1, Y+1) costs 1.
    Total = 2.
    
    Wait, the sample 1: (5,0) to (2,5).
    (5,0): j=0 (even), X = 5//2 = 2. Y=0. (2,0)
    (2,5): j=5 (odd), X = (2-1)//2 = 0. Y=5. (0,5)
    X diff = 2, Y diff = 5.
    Sample output is 5.
    If the cost was |X1-X2| + |Y1-Y2|, it would be 2 + 5 = 7.
    But it's 5. How?
    If we move from (X, Y) to (X+1, Y+1), we can potentially do it in 1 toll?
    Let's trace Sample 1: (5, 0.5) to (2.5, 5.5).
    - Move left 1: (4.5, 0.5). Cell (5,0) is tile (2,0). Cell (4,0) is tile (2,0). Toll 0.
    - Move up 1: (4.5, 1.5). Cell (4,0) is (2,0). Cell (4,1) is tile (4-1)//2 = 1. Toll 1.
    - Move left 1: (3.5, 1.5). Cell (4,1) is (1,1). Cell (3,1) is tile (3-1)//2 = 1. Toll 0.
    - Move up 3: (3.5, 4.5). Cells (3,1)->(3,2)->(3,3)->(3,4).
      (3,1) is tile (1,1).
      (3,2) is tile 3//2 = 1. Y=2. (1,2)
      (3,3) is tile (3-1)//2 = 1. Y=3. (1,3)
      (3,4) is tile 3//2 = 1. Y=4. (1,4)
      Tolls: (1,2), (1,3), (1,4) -> 3 tolls.
    - Move left 1: (2.5, 4.5). Cell (3,4) is (1,4). Cell (2,4) is 2//2 = 1. Y=4. (1,4) Toll 0.
    - Move up 1: (2.5, 5.5). Cell (2,4) is (1,4). Cell (2,5) is (2-1)//2 = 0. Y=5. (0,5) Toll 1.
    Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    Wait, look at the coordinates:
    (X1, Y1) = (2, 0)
    (X2, Y2) = (0, 5)
    The cost is 5.
    dx = 2, dy = 5.
    It seems the cost is max(dx, dy) if we can move diagonally?
    No, the move is cardinal. But moving "diagonally" in the tile-grid:
    From (X, Y) to (X+1, Y+1):
    - If we are at (X, Y), we can move to (X, Y+1) [cost 1], then to (X+1, Y+1) [cost 1]. Total 2.
    - But wait, if we move to a cell that is part of the same tile!
    In the sample, moving "left 1" often cost 0. This happens when the cell we move into is the same tile.
    This means we can change X without paying a toll if we are in a 2x1 tile.
    Specifically, for any Y, we can move between X and X+1 for free if we are in the same tile.
    
    Let's re-examine the distance:
    To move from (X1, Y1) to (X2, Y2):
    Each step in Y always costs 1 because tiles are 2x1 (horizontal).
    So the Y-distance |Y1 - Y2| is always paid.
    What about X?
    In each row Y, we can move across X-boundaries.
    