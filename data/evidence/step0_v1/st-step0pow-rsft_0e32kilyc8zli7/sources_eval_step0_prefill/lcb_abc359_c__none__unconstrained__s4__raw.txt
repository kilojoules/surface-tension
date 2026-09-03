```python
import sys

def solve():
    # Read input from standard input
    try:
        line = sys.stdin.read().split()
        if not line:
            return
        sx, sy, tx, ty = map(int, line)
    except EOFError:
        return

    # The rule for tiling is:
    # If i + j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), (4,5)... if j is even.
    # If j is odd, the pairs are (-1,0), (1,2), (3,4)...
    # Essentially, for a cell (i, j), it belongs to a tile identified by:
    # If (i + j) is even, the tile is {(i, j), (i+1, j)}
    # If (i + j) is odd, the tile is {(i, j), (i-1, j)}
    # This can be simplified: a cell (i, j) belongs to tile (floor((i + (j % 2)) / 2), j)
    
    # Let's redefine the coordinates to a system where we can use Manhattan distance.
    # The cost to move vertically is always 1 per unit because every step in y 
    # enters a new tile (since tiles are 2x1 and horizontal).
    # The cost to move horizontally depends on whether you cross a tile boundary.
    
    # Let's transform the coordinates.
    # For a fixed y, the boundaries between tiles are at x = k when (k + y) is odd.
    # If y is even, boundaries are at ..., -1, 1, 3, 5 ...
    # If y is odd, boundaries are at ..., 0, 2, 4, 6 ...
    
    # Let's use a coordinate transformation:
    # A move from (sx, sy) to (tx, ty) costs:
    # 1. The vertical distance |sy - ty|.
    # 2. The horizontal cost.
    
    # Notice that if we move from (sx, sy) to (tx, ty), we must cross |sy - ty| horizontal tiles.
    # The total cost is max(|sx - tx|, |sy - ty|) is NOT correct here because 
    # the tiles are specifically 2x1.
    
    # Let's analyze the cost:
    # To move from (sx, sy) to (tx, ty):
    # The vertical cost is |sy - ty|.
    # During these vertical moves, we can "shift" our x-position relative to the tile boundaries.
    # In each row j, the tiles are [0,1], [2,3]... if j is even, and [-1,0], [1,2]... if j is odd.
    # This is like a brick wall pattern.
    
    # Let's transform coordinates to (u, v) where:
    # u = x + (y % 2)
    # v = y
    # In the (u, v) system, the tiles are 2x1 blocks aligned to the grid.
    # Specifically, in row v, the tile boundaries are at u = ..., 0, 2, 4, ...
    # So a tile is defined by (floor(u/2), v).
    
    # The cost to move from (u1, v1) to (u2, v2):
    # Each step in v costs 1.
    # Each step in u costs 1 if it crosses a boundary (u is even).
    # However, we can move horizontally first, then vertically, then horizontally.
    # The optimal strategy is to move diagonally in the (u, v) space.
    # The distance is max(|u1 - u2|, |v1 - v2|) if we could move diagonally.
    # But we can only move L, R, U, D.
    # Actually, the cost is:
    # cost = |sy - ty| + (cost to cover remaining horizontal distance)
    # The horizontal distance is |sx - tx|.
    # In each vertical step, we can effectively move 1 unit of x for "free" 
    # (by picking the right direction) because the tile boundaries shift.
    
    # Let's use the transformation:
    # X = sx + (sy % 2), Y = sy
    # X' = tx + (ty % 2), Y' = ty
    # The distance is max(|X - X'|, |Y - Y'|) ? 
    # Let's check Sample 1: 5 0 -> 2 5
    # X = 5 + 0 = 5, Y = 0
    # X' = 2 + (5 % 2) = 3, Y' = 5
    # max(|5-3|, |0-5|) = max(2, 5) = 5. Correct.
    
    # Sample 2: 3 1 -> 4 1
    # X = 3 + 1 = 4, Y = 1
    # X' = 4 + 1 = 5, Y' = 1
    # max(|4-5|, |1-1|) = max(1, 0) = 1. 
    # Wait, Sample 2 output is 0. 
    # In Sample 2, (3,1) and (4,1) are in the same tile.
    # i=3, j=1. i+j = 4 (even). So A_{3,1} and A_{4,1} are in the same tile.
    # My X, X' logic: X=4, X'=5. The tile index is floor(X/2).
    # floor(4/2) = 2, floor(5/2) = 2. They are the same!
    
    # So the cost is:
    # Let U1 = sx + (sy % 2), V1 = sy
    # Let U2 = tx + (ty % 2), V2 = ty
    # The cost is max(|U1 - U2| // 2, |V1 - V2|) 
    # But we must be careful with the // 2.
    # The distance in U is measured in tiles. Each tile is 2 units wide.
    # The number of boundaries crossed is ceil(|U1 - U2| / 2) if we stay in the same row.
    # But we can use vertical moves to reduce horizontal cost.
    
    # Correct logic:
    # The cost is max(abs(sy - ty), (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0) + 1) // 2)
    # No, let's use the U, V transformation and the property that 
    # we can move 1 unit of U for every 1 unit of V.
    # The cost is max(|V1 - V2|, ceil(|U1 - U2| / 2))
    # Let's test Sample 1: U1=5, V1=0; U2=3, V2=5. max(5, ceil(2/2)) = 5.
    # Sample 2: U1=4, V1=1; U2=5, V2=1. max(0, ceil(1/2)) = 1. Still 1.
    # Wait, if U1=4 and U2=5, they are in the same tile (floor(4/2)=2, floor(5/2)=2).
    # The cost to move between U1 and U2 is floor(|U1 - U2| / 2) if they are in the same tile?
    # No. If U1=4, U2=5, distance is 0. If U1=4, U2=6, distance is 1.
    # The number of boundaries between U1 and U2 is:
    # If U1 < U2: boundaries are at odd integers.
    # The number of odd integers in (U1, U2] is (U2 + 1)//2 - (U1 + 1)//2.
    
    # Let's use:
    # U1 = sx + (sy % 2)
    # U2 = tx + (ty % 2)
    # V1 = sy
    # V2 = ty
    # Dist_V = abs(V1 - V2)
    # Dist_U = (U2 + 1)//2 - (U1 + 1)//2 if U1 < U2 else (U1 + 1)//2 - (U2 + 1)//2
    # Result = max(Dist_V, Dist_U)
    
    # Test Sample 2: U1=4, U2=5. Dist_U = (5+1)//2 - (4+1)//2 = 3 - 2 = 1. Still 1.
    # Let's re-evaluate. In Sample 2, sx=3, sy=1, tx=4, ty=1.
    # i=3, j=1. i+j=4 (even). A_{3,1} and A_{4,1} are in the same tile.
    # My U transformation: U1 = 3 + 1 = 4. U2 = 4 + 1 = 5.
    # These are in the same tile because floor(4/2) == floor(5/2) == 2.
    # The number of boundaries crossed is the number of odd integers strictly between U1 and U2.
    # For U1=4, U2=5, there are no odd integers between them.
    # For U1=4, U2=6, the odd integer 5 is between them.
    
    # Let's use the property:
    # Cost = max(|sy - ty|, ceil(|(sx + (sy%2)) - (tx + (ty%2))| / 2))
    # Wait, the "ceil" part is tricky.
    # Let's use: 
    # U1 = sx + (sy % 2)
    # U2 = tx + (ty % 2)
    # The number of boundaries crossed is:
    # if U1 < U2: count odd k such that U1 < k <= U2
    # if U2 < U1: count odd k such that U2 < k <= U1
    # This is equivalent to: (max(U1, U2) + 1) // 2 - (min(U1, U2) + 1) // 2
    # But we must subtract 1 if the starting point is already "past" the boundary.
    # Actually, the simplest way:
    # The tiles in row j are [0, 2), [2, 4)... if j is even (after shift)
    # The cost to move from U1 to U2 is the number of boundaries crossed.
    # A boundary is crossed if floor(U1/2) != floor(U2/2).
    # The number of boundaries between U1 and U2 is |floor(U1/2) - floor(U2/2)|.
    
    # Test Sample 1: U1=5, U2=3. |floor(5/2) - floor(3/2)| = |2 - 1| = 1.
    # max(5, 1) = 5. Correct.
    # Test Sample 2: U1=4, U2=5. |floor(4/2) - floor(5/2)| = |2 - 2| = 0.
    # max(0, 0) = 0. Correct.
    # Test Sample 3: 
    # sx=2552608206527595, sy=5411232866732612
    # tx=771856005518028, ty=7206210729152763
    # U1 = 2552608206527595 + 0 = 2552608206527595
    # U2 = 771856005518028 + 1 = 771856005518029
    # V1 = 5411232866732612, V2 = 7206210729152763
    # Dist_V = |7206210729152763 - 5411232866732612| = 1794977862420151
    # Dist_U = |floor(2552608206527595/2) - floor(771856005518029/2)|
    # Dist_U = |