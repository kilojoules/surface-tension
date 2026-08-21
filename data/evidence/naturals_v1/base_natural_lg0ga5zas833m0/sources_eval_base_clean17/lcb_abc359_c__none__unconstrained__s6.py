```python
import sys

def solve():
    """
    Problem Analysis:
    The tiles are 2x1 rectangles.
    Rule: If i + j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    This means for a fixed j:
    - If j is even: tiles are (0,1), (2,3), (4,5)... horizontally.
    - If j is odd: tiles are (1,2), (3,4), (5,6)... horizontally.
    
    Basically, for a square A_{i,j}, it belongs to a tile that covers:
    - {A_{i,j}, A_{i+1,j}} if i+j is even.
    - {A_{i-1,j}, A_{i,j}} if i+j is odd.
    
    Let's define the "tile coordinate" (u, v).
    A square A_{i,j} is part of tile T_{u,v}.
    If i+j is even, A_{i,j} and A_{i+1,j} are in T_{u,v}.
    If i+j is odd, A_{i,j} and A_{i-1,j} are in T_{u,v}.
    
    Actually, we can simplify this. 
    For a fixed j, the boundaries between tiles are at x = k where k+j is odd.
    So for j=0, boundaries are at x=1, 3, 5...
    For j=1, boundaries are at x=0, 2, 4...
    
    Let's transform the coordinates to a grid where each cell is a tile.
    A square (i, j) belongs to tile:
    v = j
    u = (i + j) // 2
    
    Wait, let's check:
    If j=0, i=0: i+j=0 (even). A_{0,0} and A_{1,0} are in the same tile. u = (0+0)//2 = 0.
    If j=0, i=1: i+j=1 (odd). A_{1,0} and A_{0,0} are in the same tile. u = (1+0)//2 = 0.
    If j=0, i=2: i+j=2 (even). A_{2,0} and A_{3,0} are in the same tile. u = (2+0)//2 = 1.
    If j=1, i=0: i+j=1 (odd). A_{0,1} and A_{-1,1} are in the same tile. u = (0+1)//2 = 0.
    If j=1, i=1: i+j=2 (even). A_{1,1} and A_{2,1} are in the same tile. u = (1+1)//2 = 1.
    
    So for any square (i, j), the tile index is (u, v) = ((i + j) // 2, j).
    Takahashi moves from (Sx, Sy) to (Tx, Ty).
    Start tile: u1 = (Sx + Sy) // 2, v1 = Sy
    End tile: u2 = (Tx + Ty) // 2, v2 = Ty
    
    The cost to move between adjacent tiles:
    - Moving from (u, v) to (u+1, v): This is a horizontal move. 
      Does it always cost 1?
      T_{u,v} and T_{u+1,v} are adjacent. To move from one to another, he must cross a boundary.
      Crossing a boundary into a new tile costs 1.
    - Moving from (u, v) to (u, v+1): This is a vertical move.
      T_{u,v} is the tile containing A_{i,j}. T_{u,v+1} is the tile containing A_{i,j+1}.
      These tiles always share a boundary (the line y = j+1).
      
    Wait, the cost is "Each time he enters a tile, he pays a toll of 1."
    This is equivalent to the Manhattan distance in the tile-grid, but we need to be careful.
    The distance is |u1 - u2| + |v1 - v2|.
    However, we must check if the start and end are in the same tile.
    If (u1, v1) == (u2, v2), cost is 0.
    Otherwise, the cost is the number of boundaries crossed.
    
    Let's re-verify with Sample 1:
    S = (5, 0), T = (2, 5)
    u1 = (5 + 0) // 2 = 2, v1 = 0
    u2 = (2 + 5) // 2 = 3, v2 = 5
    Dist = |2 - 3| + |0 - 5| = 1 + 5 = 6. 
    Sample 1 output is 5. Why?
    
    Ah, the "enters a tile" rule. If he is already in a tile, the first tile doesn't cost.
    But if he moves from T1 to T2, he pays 1.
    The distance is indeed the number of tiles entered.
    In a grid, the number of steps to get from (u1, v1) to (u2, v2) is |u1 - u2| + |v1 - v2|.
    For Sample 1: u1=2, v1=0; u2=3, v2=5. Dist = 6. Still not 5.
    
    Let's re-read: "Choose a direction... and a positive integer n. Move n units".
    If he moves vertically from (x, y) to (x, y+n), he might cross multiple tiles.
    Example: S=(5, 0), T=(2, 5).
    S is in A_{5,0}. 5+0=5 (odd), so A_{5,0} is paired with A_{4,0}. Tile T_{2,0}.
    T is in A_{2,5}. 2+5=7 (odd), so A_{2,5} is paired with A_{1,5}. Tile T_{3,5}.
    
    Wait, if he moves vertically, he might enter a tile that is "wider" and thus covers the x-coordinate.
    Let's look at the grid:
    j=0: [0,1], [2,3], [4,5], [6,7] ...
    j=1: [1,2], [3,4], [5,6], [7,8] ...
    j=2: [0,1], [2,3], [4,5], [6,7] ...
    
    At x = 5.5 (S_x + 0.5), for j=0, he is in tile T_{2,0} (covers x in [4,6]).
    For j=1, he is in tile T_{2,1} (covers x in [5,7]).
    For j=2, he is in tile T_{2,2} (covers x in [4,6]).
    
    Notice that for a fixed x, the tile index u = (i+j)//2 changes as j changes.
    If x = 5.5, then i = 5.
    j=0: u = (5+0)//2 = 2
    j=1: u = (5+1)//2 = 3
    j=2: u = (5+2)//2 = 3
    j=3: u = (5+3)//2 = 4
    j=4: u = (5+4)//2 = 4
    j=5: u = (5+5)//2 = 5
    
    Actually, the simplest way to think about this is:
    The cost is the distance in the graph where nodes are tiles and edges connect tiles that share a boundary.
    Two tiles are adjacent if:
    1. They are in the same row (v) and |u1 - u2| = 1.
    2. They are in the same column (u) and |v1 - v2| = 1.
    Wait, is (u, v) adjacent to (u, v+1)?
    Tile (u, v) contains squares A_{i,j} and A_{i+1,j} (if i+j even).
    Tile (u, v+1) contains squares A_{k,j+1} and A_{k+1,j+1} (if k+j+1 even).
    They are adjacent if they share a boundary segment.
    The boundary is the line y = j+1.
    Tile (u, v) covers x in [i, i+2].
    Tile (u, v+1) covers x in [k, k+2].
    They share a boundary if [i, i+2] and [k, k+2] overlap.
    
    Let's use the coordinate transformation:
    For a point (x, y), let i = floor(x), j = floor(y).
    The tile index is u = (i + j) // 2, v = j.
    The distance between (u1, v1) and (u2, v2) in this grid is |u1 - u2| + |v1 - v2|.
    But wait, if we move from (u, v) to (u, v+1), the x-range of the tile changes.
    Tile (u, v) covers x in [2u - v, 2u - v + 2] if v is even? No.
    Let's re-calculate:
    If i+j is even, tile is {A_{i,j}, A_{i+1,j}}. u = (i+j)//2.
    The x-range is [i, i+2].
    Since i = 2u - j, the x-range is [2u - j, 2u - j + 2].
    
    Now, if we are at tile (u, v), we can move to:
    - (u+1, v) and (u-1, v) with cost 1.
    - (u, v+1) and (u, v-1). Do these always share a boundary?
      Tile (u, v) x-range: [2u - v, 2u - v + 2]
      Tile (u, v+1) x-range: [2u - (v+1), 2u - (v+1) + 2] = [2u - v - 1, 2u - v + 1]
      The intersection is [2u - v, 2u - v + 1], which has length 1.
      Yes, they always share a boundary.
      
    So the distance is simply |u1 - u2| + |v1 - v2|.
    Let's check Sample 1 again:
    S = (5, 0) -> i=5, j=0. u1 = (5+0)//2 = 2, v1 = 0.
    T = (2, 5) -> i=2, j=5. u2 = (2+5)//2 = 3, v2 = 5.
    Dist = |2 - 3| + |0 - 5| = 1 + 5 = 6.
    Still 6. What is wrong?
    
    Wait! "Each time he enters a tile, he pays a toll of 1."
    If he starts in tile T_start and ends in tile T_end, he pays for every tile entered *after* the first one.
    So the cost is the number of edges in the shortest path.
    But the sample says 5. Let's re-read.
    "Move left by 1. Pay a toll of 0."
    S = (5.5, 0.5). i=5, j=0. u1 = (5+0)//2 = 2.
    Move left by 1: (4.5, 0.5). i=4, j=0. u = (4+0)//2 = 2.
    He is still in the same tile! That's why the toll is 0.
    
    Then "Move up by 1. Pay a toll of 1."
    (4.5, 1.5). i=4, j=1. u = (4+1)//2 = 2.
    He entered tile (2, 1). Cost = 1.
    
    Then "Move left by 1. Pay a toll of 0."
    (3.5, 1.5). i=3, j=1. u = (3+1)//2 = 2.
    Still in tile (2, 1).
    
    Then "Move up by 3. Pay a toll of 3."
    (3.5, 4.5). i=3, j=4. u = (3+