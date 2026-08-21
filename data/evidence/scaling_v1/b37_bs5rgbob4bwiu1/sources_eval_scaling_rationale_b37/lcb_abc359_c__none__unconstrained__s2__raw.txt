The constraint to avoid explicit loops and recursion forces a shift from imperative state management to a functional style, emphasizing expressions over statements. This encourages the use of declarative constructs that describe *what* the result is rather than *how* to step through the calculation, leveraging Python's powerful comprehension syntax and built-in mathematical functions to handle coordinate transformations and distance calculations in a single flow. For this problem, the core logic involves mapping the staggered tile grid into a coordinate system where movement costs are uniform, allowing the minimum toll to be calculated using a direct formula based on the Manhattan distance between the transformed start and end points.

```python
import sys

def solve():
    # Read input and parse coordinates
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling rule: 
    # If i+j is even, (i, j) and (i+1, j) are in the same tile.
    # This means tiles are 2x1 horizontal blocks starting at (i, j) where i+j is even.
    # Let's transform the coordinates to a system where each unit of distance 
    # represents a boundary crossing that costs a toll.
    
    # In the original grid:
    # Moving vertically always crosses a tile boundary (cost 1).
    # Moving horizontally crosses a boundary every 1 unit, 
    # UNLESS you are moving within a 2x1 tile.
    # A tile exists at {(i, j), (i+1, j)} if i+j is even.
    
    # Transformation:
    # Let X = x + y
    # Let Y = y
    # However, a simpler approach is to realize that the cost to move from 
    # (sx, sy) to (tx, ty) is related to the number of horizontal and vertical 
    # boundaries crossed.
    
    # Let's redefine coordinates to simplify the "staggered" nature:
    # Each row y has boundaries at x = k if (k + y) is odd.
    # The distance is the sum of vertical steps and horizontal steps 
    # that actually cross a tile boundary.
    
    # Let f(x, y) be a transformation to a coordinate system where 
    # the distance is the answer.
    # For a fixed y, the boundaries are at x such that x+y is odd.
    # This is equivalent to saying boundaries are at x = 2k + 1 if y is even
    # and x = 2k if y is odd.
    
    # The cost to move from (sx, sy) to (tx, ty):
    # Vertical cost: |sy - ty|
    # Horizontal cost: 
    # To move from sx to tx at height sy, we cross boundaries.
    # The boundaries are at x where x+sy is odd.
    # The number of such x between sx and tx is the number of integers k 
    # such that min(sx, tx) < k <= max(sx, tx) and k+sy is odd.
    
    # This is a bit complex because we can change y to find a "cheaper" x-path.
    # Actually, the optimal strategy is:
    # 1. Move vertically to ty.
    # 2. Move horizontally.
    # But we can move horizontally at any y.
    # At height y, the cost to move from sx to tx is the number of k in (sx, tx] 
    # such that k+y is odd. This is either floor(|sx-tx|/2) or ceil(|sx-tx|/2).
    # We can pick y to minimize this.
    
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the boundaries are at i where i+j is odd.
    # To get from sx to tx, we must cross boundaries. 
    # In any row j, the number of boundaries between sx and tx is:
    # If sx < tx: count k in {sx+1, ..., tx} such that k+j is odd.
    # If tx < sx: count k in {tx+1, ..., sx} such that k+j is odd.
    
    # Let dx = abs(sx - tx)
    # The number of boundaries crossed is either dx // 2 or (dx + 1) // 2.
    # We can choose j (by moving up/down) to pick the minimum, which is dx // 2.
    # However, we must also account for the vertical distance dy = abs(sy - ty).
    # Each vertical step costs 1.
    
    # Total cost = dy + (dx // 2) is almost correct, but we need to check if
    # we can optimize by moving to a row where the horizontal crossing is cheaper.
    # But moving to another row costs vertical tolls.
    
    # Correct Logic:
    # Let's transform coordinates:
    # X = x + y
    # Y = y
    # This doesn't quite work. Let's use the property:
    # Cost = abs(sy - ty) + (abs(sx - tx) + 1) // 2 is NOT correct.
    # The actual minimum toll is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # If we stay in the same row, cost is (dx + 1) // 2 if the boundaries align poorly.
    # But we can move to an adjacent row to potentially save 1 unit of horizontal cost
    # at the expense of 2 units of vertical cost. That's not helpful.
    
    # The only way to reduce horizontal cost is to be in a row where 
    # the boundaries are shifted.
    # For a fixed y, the cost to go from sx to tx is:
    # count k in (min(sx, tx), max(sx, tx)] such that k+y is odd.
    # This count is dx // 2 if (sx+y) and (tx+y) have the same parity and 
    # the range is "lucky", or (dx+1)//2 otherwise.
    # Specifically, the count is dx // 2 if (sx+y)%2 == (tx+y)%2 and 
    # the boundaries are not hit, etc.
    # Actually, the number of k in (sx, tx] such that k+y is odd is:
    # (tx - sx + 1) // 2 if we pick the right y.
    
    # Let''s use the transformation:
    # New X = x + y
    # New Y = y
    # This is still not quite right. Let's use:
    # The distance is abs(sy - ty) + (abs(sx - tx) + 1) // 2 
    # EXCEPT when we can save a toll by moving horizontally in a row where 
    # the start and end points are in the same tile or boundaries are shifted.
    
    # Let's use the coordinate transform:
    # u = x + y
    # v = y
    # The cost is max(abs(u1 - u2), abs(v1 - v2))? No.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # Vertical distance dy = abs(sy - ty).
    # Horizontal distance dx = abs(sx - tx).
    # In any row y, the number of boundaries crossed is:
    # floor((dx + 1) / 2) if (sx+y)%2 != (tx+y)%2
    # floor(dx / 2) if (sx+y)%2 == (tx+y)%2 AND we are lucky...
    
    # Let's use the property:
    # The cost is abs(sy - ty) + (abs(sx - tx) + 1) // 2
    # BUT if (sx + sy) % 2 == (tx + ty) % 2, we might save 1.
    # Wait, the simplest transformation is:
    # X = x + y
    # Y = y
    # Then the distance is max(abs(X1-X2), abs(Y1-Y2))? No.
    
    # Let's use the logic from similar problems:
    # The answer is abs(sy - ty) + (abs(sx - tx) + 1) // 2
    # if (sx + sy) % 2 != (tx + ty) % 2 else ...
    
    # Correct transformation for this specific grid:
    # Let x' = x + y, y' = y
    # The distance is abs(y1 - y2) + (abs(x1 - x2) + 1) // 2 is wrong.
    # Let's use:
    # ans = abs(sy - ty) + (abs(sx - tx) + 1) // 2
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 0:
    # This is not helping.
    
    # Let's use the most reliable transformation for this problem:
    # X = x + y
    # Y = y
    # The distance is abs(Y1 - Y2) + (abs(X1 - X2) + 1) // 2 
    # is still not it.
    
    # Final attempt at logic:
    # The cost is abs(sy - ty) + (abs(sx - tx) + 1) // 2
    # However, if (sx + sy) % 2 == 0 and (tx + ty) % 2 == 0, 
    # and we move horizontally in a row where we don't cross a boundary 
    # at the start/end, we can save.
    
    # Actually, the simplest correct answer is:
    # X1 = sx + sy, Y1 = sy
    # X2 = tx + ty, Y2 = ty
    # Ans = max(abs(X1 - X2), abs(Y1 - Y2))
    # Let's test Sample 1: 5 0, 2 5 -> X1=5, Y1=0; X2=7, Y2=5. max(2, 5) = 5. Correct.
    # Sample 2: 3 1, 4 1 -> X1=4, Y1=1; X2=5, Y2=1. max(1, 0) = 1. Incorrect (Sample says 0).
    
    # Sample 2 again: 3 1, 4 1. sx=3, sy=1, tx=4, ty=1.
    # i+j = 3+1 = 4 (even). So A_{3,1} and A_{4,1} are the same tile.
    # Cost is 0.
    # My X, Y: X1=4, Y1=1; X2=5, Y2=1. max(1, 0) = 1.
    
    # Let's try: X = x, Y = y. 
    # If i+j is even, (i,j) and (i+1,j) are one tile.
    # This means boundaries are at i+1/2 where i+j is odd.
    # Let's use: X = x + (y % 2), Y = y
    # Sample 2: sx=3, sy=1, tx=4, ty=1.
    # X1 = 3 + 1 = 4, Y1 = 1
    # X2 = 4 + 1 = 5, Y2 = 1
    # Still not 0.
    
    # The correct transformation is:
    # X = x + (y & 1)
    # Y = y
    # Wait, the rule is i+j is even.
    # If y is even, boundary is at i where i is odd.
    # If y is odd, boundary is at i where i is even.
    # This means the boundary is always at i where i % 2 != y % 2.
    # Let's use: X = x + (y % 2), Y = y. No.
    
    # Let's use:
    # If y is even, tiles are [0,1], [2,3], [4,5]...
    # If y is odd, tiles are [-1,0], [1,2], [3,4]...
    # This means in row y, the tile index is (x + (y % 2)) // 2.
    # Let tx_tile = (tx + (ty % 2)) // 2
    # Let ty_tile = ty
    # Let sx_tile = (sx + (sy % 2)) // 2
    # Let sy_tile = sy
    # Distance = abs(sx_tile - tx_tile) + abs(sy_tile - ty_tile)
    # Sample 1: 5 0, 2 5
    # sx