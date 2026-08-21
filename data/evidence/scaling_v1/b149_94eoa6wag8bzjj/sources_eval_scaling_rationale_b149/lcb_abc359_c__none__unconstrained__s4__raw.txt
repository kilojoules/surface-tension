The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing the transformation of data. For this problem, the optimal path is determined by the Manhattan distance and the specific layout of the $2 \times 1$ tiles, which allows for "free" horizontal movement depending on the parity of the coordinates. I will use `abs()` for distances and `max()`/`min()` to handle the conditional logic of the tile boundaries.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1. 
    # A_{i,j} and A_{i+1,j} are the same tile if i+j is even.
    # This means for a fixed j, tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    
    # Let's normalize the coordinates to a 2x2 grid system.
    # A tile covers two squares. We can think of the plane as being divided into
    # 2x2 blocks. In each 2x2 block:
    # Row 0 (y=0): Square (0,0) and (1,0) are one tile.
    # Row 1 (y=1): Square (1,1) and (2,1) are one tile.
    
    # The cost to move between two points in this specific tiling is:
    # cost = max(|sx - tx|, |sy - ty|) if we could move diagonally, 
    # but we move orthogonally.
    # Actually, the cost is simply the Manhattan distance in a transformed coordinate system.
    # Let's map (x, y) to a coordinate system where each unit is a tile.
    # For a fixed y, the tiles are grouped in pairs.
    # If y is even, tiles are {(0,0), (1,0)}, {(2,0), (3,0)}... -> x' = x // 2
    # If y is odd, tiles are {(-1,1), (0,1)}, {(1,1), (2,1)}... -> x' = (x + 1) // 2
    
    # However, a simpler observation:
    # To move from (sx, sy) to (tx, ty), the minimum cost is:
    # ceil(|sx - tx| / 2) + |sy - ty| is NOT correct because we can move 
    # horizontally for free within a tile.
    # The correct distance is:
    # Let dx = |sx - tx| and dy = |sy - ty|
    # The cost is dy + max(0, (dx - (1 if (sx+sy)%2 == (tx+sy)%2 and 
    #                                 ((sx//2 == tx//2 if sy%2==0 else (sx+1)//2 == (tx+1)//2)) 
    #                                 else 0)) // 2)
    # Actually, the most reliable way to view this is:
    # You pay 1 for every vertical step.
    # You pay 1 for every 2 horizontal steps, but the "free" step 
    # depends on the parity of the current row.
    
    # Let's use the transformation:
    # A point (x, y) belongs to tile ( (x if y%2==0 else x+1)//2, y )
    # Let X(x, y) = (x if y%2==0 else x+1) // 2
    # Let Y(x, y) = y
    # The distance is |X(sx, sy) - X(tx, ty)| + |Y(sx, sy) - Y(tx, ty)|
    # But wait, you can change your X coordinate "for free" by moving vertically 
    # into a tile that covers your target X.
    
    # The minimum cost is:
    # cost = |sy - ty| + max(0, (|X(sx, sy) - X(tx, ty)| - 1)) 
    # This is still not quite right. 
    # The correct logic:
    # The cost is |sy - ty| + max(0, abs(sx - tx) - 1) // 2 
    # if we can leverage the tile widths.
    # But the tiles shift every row.
    # The distance is simply:
    # dist = max(|sx - tx|, |sy - ty|, (|sx - tx| + |sy - ty| + 1) // 2) 
    # No, that's for different problems.
    
    # Correct logic for this specific tiling:
    # The cost is |sy - ty| + max(0, (abs(sx - tx) - (1 if (sx+sy)%2 == 0 and (tx+sy)%2 == 0 and sx//2 != tx//2 else 0)) // 2)
    # Let's use the property: cost = max(|sy - ty|, (|sx - tx| + |sy - ty|) // 2)
    # Wait, the simplest form is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2)
    # Let's test Sample 1: 5 0, 2 5 -> max(5, (3+5)//2) = 5. Correct.
    # Sample 2: 3 1, 4 1 -> max(0, (1+0)//2) = 0. Correct.
    # Sample 3: max(1843987936321, (17754224072097 + 1843987936321)//2) = 8969311426259
    # Wait, Sample 3 output is 1794977862420151. My formula is wrong.
    
    # Let's re-evaluate.
    # In row y, tiles are [2k, 2k+1] if y is even, [2k-1, 2k] if y is odd.
    # This is a grid where you can move 2 units horizontally for the price of 1 
    # (by moving Y -> Y+1 -> Y), but you can't move 2 units horizontally for free.
    # Actually, the cost is:
    # cost = |sy - ty| + max(0, (abs(sx - tx) - 1) // 2) if we can't use the 
    # current tile to cover the distance.
    
    # Let's use the coordinate transformation:
    # Each tile can be identified by (u, v) where v = y and u = (x + (y%2)) // 2.
    # Moving from (u1, v1) to (u2, v2):
    # A move in Y changes v by 1 and may change u by 0 or 1.
    # A move in X changes u by 1 and costs 1 (since it crosses a tile boundary).
    # But moving X by 1 might be free if you stay in the same tile.
    # The distance is:
    # cost = abs(v1 - v2) + max(0, abs(u1 - u2) - 1) 
    # Wait, if we move vertically, we can change our 'u' for free if we 
    # pick the right boundary.
    # The distance is actually:
    # cost = max(abs(v1 - v2), abs(u1 - u2))
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5
    # u1 = (5 + 0)//2 = 2, v1 = 0
    # u2 = (2 + 1)//2 = 1, v2 = 5
    # max(|0-5|, |2-1|) = 5. Correct.
    # Sample 2: sx=3, sy=1, tx=4, ty=1
    # u1 = (3 + 1)//2 = 2, v1 = 1
    # u2 = (4 + 1)//2 = 2, v2 = 1
    # max(0, 0) = 0. Correct.
    # Sample 3: sx=2552608206527595, sy=5411232866732612, tx=771856005518028, ty=7206210729152763
    # u1 = (2552608206527595 + 0)//2 = 1276304103263797
    # v1 = 5411232866732612
    # u2 = (771856005518028 + 1)//2 = 385928002759014
    # v2 = 7206210729152763
    # max(|5411232866732612 - 7206210729152763|, |1276304103263797 - 385928002759014|)
    # max(794977862410151, 890376100504783) = 890376100504783.
    # Still not 1794977862420151.
    
    # Let's reconsider: the cost is simply the Manhattan distance in the (u, v) space?
    # cost = abs(u1 - u2) + abs(v1 - v2)
    # Sample 1: |2-1| + |0-5| = 1 + 5 = 6. (Wrong, should be 5)
    # Sample 3: 890376100504783 + 794977862410151 = 1685353962914934. (Wrong)
    
    # Final attempt at logic:
    # The distance is abs(sy - ty) + max(0, abs(sx - tx) - 1) // 2 
    # is only if you can't move diagonally. But you can't.
    # However, you can move 1 unit Y and 1 unit X and only pay for the Y.
    # The cost is: abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Sample 1: 5 + max(0, (3 - 5 + 1)//2) = 5 + 0 = 5.
    # Sample 3: 794977862410151 + max(0, (1775422406527595 - 794977862410151 + 1)//2)
    # = 794977862410151 + (980444544117445)//2 = 794977862410151 + 490222272058722 = 1285199934468873.
    # Still not it. The only way to get 1794977862420151 is:
    # (1775422406527595 + 794977862410151 + 1) // 2 = 1285199934468873.
    # Wait: (1775422406527595 + 794977862410151) // 2 is not it.
    # What if the cost is: (abs(sx - tx) + abs(sy - ty) + 1) // 2?
    # Sample 1: (3 + 5 + 1) // 2 = 4. (Wrong)
    # Let's try: cost = abs(sy - ty) + (abs(sx - tx) + 1