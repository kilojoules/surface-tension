```python
import sys

def solve():
    """
    The problem describes a tiling of the plane with 2x1 tiles.
    Rule: A_{i,j} and A_{i+1,j} are in the same tile if i+j is even.
    
    Let's analyze the tiles:
    A tile is a 2x1 block.
    If i+j is even, the tile covers {(i, j), (i+1, j)}.
    If i+j is odd, the tile must cover A_{i,j} and something else. 
    Since A_{i-1,j} is paired with A_{i,j} if (i-1)+j is even (which it is), 
    A_{i,j} is paired with A_{i-1,j} when i+j is odd.
    
    Wait, let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    This means:
    For a fixed j:
    - If j is even: (0,j)&(1,j) are together, (2,j)&(3,j) are together, ...
    - If j is odd: (1,j)&(2,j) are together, (3,j)&(4,j) are together, ...
    
    Basically, for a row j, the tiles are horizontal 2x1 blocks.
    The boundary between tiles in row j is at x = k where k+j is odd.
    
    Takahashi starts at (Sx+0.5, Sy+0.5) and ends at (Tx+0.5, Ty+0.5).
    Moving within a tile costs 0. Entering a new tile costs 1.
    
    Let's map the coordinates to a grid where each node is a tile.
    A cell (x, y) belongs to tile:
    If (x + y) is even, it's the left half of a tile: TileID = ( (x >> 1) if y even else (x-1)//2, y ) ? 
    Actually, simpler:
    In row y, the tiles are:
    If y is even: [0,1], [2,3], [4,5] ... 
    If y is odd:  [-1,0], [1,2], [3,4] ...
    
    Let's transform the coordinates (x, y) to (u, v) such that moving from (u, v) to (u', v') 
    costs distance.
    
    Notice the pattern:
    The distance is essentially the Manhattan distance in a transformed space.
    Let's look at the cost to move from (Sx, Sy) to (Tx, Ty).
    The vertical distance is simply |Sy - Ty|. Each vertical step enters a new tile.
    The horizontal distance:
    In row y, the tiles are boundaries at x = k where k+y is odd.
    The number of boundaries crossed is the number of k such that min(Sx, Tx) < k <= max(Sx, Tx) and k+y is odd.
    
    However, he can change rows.
    Let's define a coordinate system (u, v) where:
    v = y
    u = x if y is even else x + 0.5 (effectively)
    
    Actually, the problem is equivalent to finding the shortest path on a graph.
    The cost is:
    Let dx = abs(Sx - Tx)
    Let dy = abs(Sy - Ty)
    
    If we are at (x, y) and move to (x, y+1), we always enter a new tile.
    If we are at (x, y) and move to (x+1, y), we enter a new tile IF the boundary is between x and x+1.
    The boundary is at x+1 if (x+1)+y is odd.
    
    Let's transform:
    Let X = Sx, Y = Sy
    Let X' = Tx, Y' = Ty
    
    The distance is:
    dist = abs(Y - Y') + max(0, abs(X - X') - 1) is not quite right.
    
    Correct logic for this specific tiling:
    The cost to move from (x, y) to (x', y') is:
    Let dx = abs(x - x')
    Let dy = abs(y - y')
    The answer is (dx + dy) // 2 * 2 ... no.
    
    Let's use the property:
    The cost is the Manhattan distance in a coordinate system where we move by 2 units to cost 1.
    Actually, the formula for this specific grid is:
    ans = abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy)//2 - (Tx - Ty)//2 ) 
    No, that's for a different problem.
    
    Let's re-evaluate:
    Cost to move from (x, y) to (x+1, y): 1 if x+y is even, 0 if x+y is odd.
    Cost to move from (x, y) to (x, y+1): 1 always.
    
    Wait, if we move (x, y) -> (x, y+1) -> (x+1, y+1), the total cost is 1 + (1 if x+y+1 is even else 0).
    If we move (x, y) -> (x+1, y) -> (x+1, y+1), the total cost is (1 if x+y is even else 0) + 1.
    
    This looks like the distance is:
    Let f(x, y) = (x + y) // 2
    The distance is abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy)//2 - (Tx - Ty)//2 ) is for 45-degree rotated.
    
    Let's try:
    Let u = (x + y) // 2
    Let v = (x - y) // 2
    This doesn't seem right.
    
    Let's use the property:
    The cost is simply:
    abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy)//2 - (Tx - Ty)//2 ) 
    Wait, let's test Sample 1: 5 0 and 2 5.
    Sx=5, Sy=0 -> (5+0)//2 = 2, (5-0)//2 = 2
    Tx=2, Ty=5 -> (2+5)//2 = 3, (2-5)//2 = -2 (integer division in python: -3)
    abs(2 - 3) + abs(2 - (-3)) = 1 + 5 = 6. Sample says 5.
    
    Let's try another approach:
    The cost is:
    abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy + 1)//2 - (Tx - Ty + 1)//2 )
    No.
    
    Correct logic:
    The distance is:
    abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy)//2 - (Tx - Ty)//2 ) is for a different tiling.
    
    Let's look at the grid again.
    The tiles are 2x1.
    For row y, tiles are [2k, 2k+1] if y is even, [2k-1, 2k] if y is odd.
    This means a tile is defined by ( (x + (y%2)) // 2, y ).
    Let's call this (u, v) where u = (x + (y%2)) // 2 and v = y.
    Moving from (u, v) to (u', v'):
    - If v = v', cost is abs(u - u').
    - If v != v', we move to (u, v+1). The tile at (u, v+1) is ((x + ((v+1)%2)) // 2, v+1).
    
    Let's simplify:
    The distance is abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy)//2 - (Tx - Ty)//2 ) is for a grid rotated by 45 degrees.
    Actually, for this specific problem, the answer is:
    abs( (Sx + Sy)//2 - (Tx + Ty)//2 ) + abs( (Sx - Sy + 1)//2 - (Tx - Ty + 1)//2 ) ?
    
    Let's try:
    Sx=5, Sy=0; Tx=2, Ty=5
    u1 = (5+0)//2 = 2, v1 = (5-0+1)//2 = 3
    u2 = (2+5)//2 = 3, v2 = (2-5+1)//2 = -1
    abs(2-3) + abs(3 - (-1)) = 1 + 4 = 5. Correct!
    
    Sample 2: 3 1 and 4 1
    Sx=3, Sy=1; Tx=4, Ty=1
    u1 = (3+1)//2 = 2, v1 = (3-1+1)//2 = 1
    u2 = (4+1)//2 = 2, v2 = (4-1+1)//2 = 2
    abs(2-2) + abs(1-2) = 1. Sample says 0.
    
    Wait, Sample 2: 3 1 and 4 1.
    S=(3,1), T=(4,1).
    S is in tile ( (3 + (1%2))//2, 1 ) = ( (3+1)//2, 1 ) = (2, 1).
    T is in tile ( (4 + (1%2))//2, 1 ) = ( (4+1)//2, 1 ) = (2, 1).
    They are in the same tile! Cost 0.
    
    My u, v formula:
    u = (x + y) // 2
    v = (x - y) // 2
    
    Let's check (3,1) and (4,1) with u = (x+y)//2, v = (x-y)//2:
    (3+1)//2 = 2, (3-1)//2 = 1
    (4+1)//2 = 2, (4-1)//2 = 1
    abs(2-2) + abs(1-1) = 0. Correct!
    
    Let's check Sample 1 again with u = (x+y)//2, v = (x-y)//2:
    (5,0) -> (5+0)//2 = 2, (5-0)//2 = 2
    (2,5) -> (2+5)//2 = 3, (2-5)//2 = -2 (integer division: -2)
    abs(2-3) + abs(2 - (-2)) = 1 + 4 = 5. Correct!
    
    Sample 3:
    2552608206527595 5411232866732612
    771856005518028 7206210729152763
    S: x=2552608206527595, y=5411232866732612
    T: x=771856005518028, y=7206210729152763
    u1 = (2552608206527595 + 5411232866732612) // 2 = 7963841073260207 // 2 = 3981920536630103
    v1 = (2552608206527595 - 5411232866732612) // 2 = -2858624660205017 // 2 = -1429312330102509
    u2 = (771856005518028 + 7206210729152763) // 2 = 7978066734670791 // 2 = 3989033367335395
    v2 = (7718560055