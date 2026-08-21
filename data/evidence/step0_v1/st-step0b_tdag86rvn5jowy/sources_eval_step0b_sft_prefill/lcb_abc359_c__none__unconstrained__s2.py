```python
import sys

def solve():
    # Read input from stdin
    # The input consists of Sx, Sy, Tx, Ty
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are horizontal 2x1 blocks.
    # In row j, the boundaries between tiles are at x = k where (k + j) is odd.
    # To move from (sx, sy) to (tx, ty), we want to minimize the number of tiles entered.
    
    # Let's transform the coordinates to a system where the grid is uniform.
    # Consider the transformation: 
    # u = x + y
    # v = x - y
    # However, a simpler way is to observe the Manhattan distance and the parity.
    # The cost to move between two adjacent squares (i, j) and (i', j') is:
    # 0 if they belong to the same tile, 1 otherwise.
    
    # Let f(i, j) be the coordinates of the tile containing square (i, j).
    # If (i + j) is even, square (i, j) and (i+1, j) are in tile ( (i//2)*2, j ).
    # If (i + j) is odd, square (i, j) and (i-1, j) are in tile ( ((i-1)//2)*2, j ).
    # Actually, a simpler representation:
    # Square (i, j) belongs to tile ( (i + (i+j)%2) // 2, j ) if we consider the 2x1 blocks.
    
    # Let's use the property:
    # To move from (sx, sy) to (tx, ty), the minimum cost is:
    # ceil( (abs(sx - tx) + abs(sy - ty)) / 2 )
    # But we must check if the start and end squares are in the same tile and 
    # if the path can be optimized.
    
    # Correct logic for this specific tiling:
    # The distance is (abs(sx - tx) + abs(sy - ty)) // 2.
    # We need to account for the "phase" of the tiles.
    # Let g(x, y) = (x + y) % 2.
    # If we move 1 unit vertically, we always enter a new tile.
    # If we move 1 unit horizontally, we enter a new tile only if (x + y) is odd.
    
    # The minimum cost is actually:
    # (abs(sx - tx) + abs(sy - ty) + 1) // 2
    # But we must subtract 1 if the start and end points are in the same tile.
    # Wait, the sample 1: (5,0) to (2,5). abs(5-2) + abs(0-5) = 3 + 5 = 8. 8 // 2 = 4.
    # Sample 1 output is 5.
    
    # Let's re-evaluate.
    # Let x' = sx, y' = sy, x'' = tx, y'' = ty.
    # The cost is (abs(x' - x'') + abs(y' - y'')) // 2.
    # If (x' + y') % 2 != (x'' + y'') % 2, the cost is (abs(x' - x'') + abs(y' - y'')) // 2 + 1.
    # No, that's for a different problem.
    
    # Let's use the coordinate transformation:
    # Each tile can be identified by (X, Y) where X = (i + (i+j)%2)//2 and Y = j.
    # The distance between (X1, Y1) and (X2, Y2) in this specific graph is:
    # abs(X1 - X2) + abs(Y1 - Y2).
    # But we can move diagonally in the (X, Y) space by moving 1 unit in x and 1 unit in y.
    # Specifically, moving from (i, j) to (i, j+1) costs 1.
    # Moving from (i, j) to (i+1, j) costs 0 or 1.
    
    # The actual minimum cost is:
    # abs(sy - ty) + (abs(sx - tx) + 1) // 2
    # But we can optimize by zigzagging.
    # The correct formula for this specific tiling problem is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's test Sample 1: max(5, (3 + 5 + 1) // 2) = max(5, 4) = 5. Correct.
    # Sample 2: max(0, (1 + 0 + 1) // 2) = max(0, 1) = 1. Incorrect. Sample 2 is 0.
    
    # Re-evaluating Sample 2: (3, 1) to (4, 1).
    # i=3, j=1 => i+j=4 (even). A_{3,1} and A_{4,1} are in the same tile.
    # Cost is 0.
    
    # The logic is:
    # To move from (sx, sy) to (tx, ty), we can move in "diagonal" steps.
    # A step of (1, 1) or (1, -1) costs 1 toll (entering the new row's tile).
    # A step of (2, 0) costs 1 toll.
    # A step of (1, 0) costs 0 or 1 toll.
    
    # Let dx = abs(sx - tx), dy = abs(sy - ty).
    # We can cover 2 units of dx and 1 unit of dy with 1 toll (by moving 1 right, 1 up).
    # No, the most efficient way to move is to use the 2x1 tiles.
    # In each row, we can move 2 units horizontally for 1 toll.
    # But we can also move 1 unit vertically for 1 toll.
    # If we move (1, 1), we enter a new tile.
    # The minimum cost is:
    # If we are at (sx, sy) and want to reach (tx, ty):
    # We can move dy units vertically (cost dy).
    # This leaves us with dx units to cover horizontally.
    # Some of these dx units might have been covered "for free" if we timed our vertical moves.
    # Actually, the distance is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: 5 + max(0, (3 - 5 + 1) // 2) = 5 + 0 = 5. Correct.
    # Sample 2: 0 + max(0, (1 - 0 + 1) // 2) = 1. Still 1.
    
    # Wait, Sample 2: (3, 1) and (4, 1). i=3, j=1. i+j=4 (even).
    # Rule: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    # My formula gives 1 because it doesn't know which specific x we are at.
    
    # Let's use the property:
    # Square (i, j) is in tile ( (i + (i+j)%2)//2, j ).
    # Let X(i, j) = (i + (i+j)%2)//2 and Y(i, j) = j.
    # Moving from (i, j) to (i+1, j) is 0 cost if (i+j) is even, 1 if odd.
    # Moving from (i, j) to (i, j+1) is always 1 cost.
    
    # The distance is:
    # abs(Y1 - Y2) + (abs(X1 - X2) if we can't optimize)
    # Actually, the distance is simply:
    # abs(Y1 - Y2) + max(0, abs(X1 - X2) - abs(Y1 - Y2))
    # No, that's just abs(X1 - X2) + abs(Y1 - Y2) if X is larger.
    
    # Let's use the property:
    # To go from (X1, Y1) to (X2, Y2), we can use "diagonal" moves.
    # A move (X, Y) -> (X+1, Y+1) is possible by moving (i, j) -> (i+1, j) -> (i+1, j+1).
    # If (i+j) is even, (i, j) and (i+1, j) are one tile. Then moving to (i+1, j+1) costs 1.
    # So (X, Y) -> (X+1, Y+1) costs 1.
    # Similarly (X, Y) -> (X-1, Y+1) costs 1.
    # (X, Y) -> (X, Y+1) costs 1.
    # (X, Y) -> (X+1, Y) costs 1.
    
    # This is a distance metric: dist = max(abs(X1 - X2), abs(Y1 - Y2))
    # Let's check Sample 1:
    # S: (5, 0) -> X = (5 + (5+0)%2)//2 = (5+1)//2 = 3, Y = 0.
    # T: (2, 5) -> X = (2 + (2+5)%2)//2 = (2+1)//2 = 1, Y = 5.
    # max(abs(3-1), abs(0-5)) = max(2, 5) = 5. Correct.
    # Sample 2:
    # S: (3, 1) -> X = (3 + (3+1)%2)//2 = (3+0)//2 = 1, Y = 1.
    # T: (4, 1) -> X = (4 + (4+1)%2)//2 = (4+1)//2 = 2, Y = 1.
    # Wait, (3,1) and (4,1) are in the same tile.
    # Let's re-calculate X for (3,1): (3 + (3+1)%2)//2 = (3+0)//2 = 1.
    # For (4,1): (4 + (4+1)%2)//2 = (4+1)//2 = 2.
    # Still different. Let me re-read.
    # "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # i=3, j=1 => i+j=4 (even). So A_{3,1} and A_{4,1} are the same tile.
    # My X formula: X(3,1) = (3+0)//2 = 1. X(4,1) = (4+1)//2 = 2.
    # The formula should be: if (i+j)%2 == 0, then i and i+1 are grouped.
    # This means the groups are {0,1}, {2,3}, {4,5} when j is even, and {1,2}, {3,4}, {5,6} when j is odd.
    # This is exactly what happens if we shift i by (j%2).
    # Let i' = i + (j%2). Then the tiles are (i' // 2, j).
    # Sample 2: S(3, 1) -> i'=3+1=4, X=2, Y=1. T(4, 1) -> i'=4+1=5, X=2, Y=1.
    # max(abs(2-2), abs(1-1)) = 0. Correct!
    
    # Let's check Sample 1 again:
    # S(5, 0) -> i'=5+0=5, X=2, Y=0.
    # T(2, 5) -> i'=2+1=3, X=1, Y=5.
    # max(abs(2-1), abs