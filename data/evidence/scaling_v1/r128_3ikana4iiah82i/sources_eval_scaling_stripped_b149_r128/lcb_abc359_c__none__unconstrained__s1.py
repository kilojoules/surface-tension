```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern repeats every 2 units in x and y.
    # We can map the coordinates to a smaller range while preserving the 
    # parity-based tile structure.
    # However, the distance is the primary factor.
    # Let's analyze the cost:
    # Moving in Y always crosses a tile boundary every 1 unit.
    # Moving in X crosses a boundary every 1 unit, UNLESS the two 
    # squares are part of the same 2x1 tile.
    # A tile consists of A_{i,j} and A_{i+1,j} if i+j is even.
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # 1. The vertical distance |sy - ty|.
    # 2. The horizontal distance |sx - tx|, but we can skip tolls 
    #    if we move across the boundary of a 2x1 tile.
    
    # Specifically, if we are at (x, y) and move to (x+1, y), 
    # the toll is 0 if x+y is even, and 1 if x+y is odd.
    # This is equivalent to saying we pay 1 toll for every 2 units of X,
    # depending on the starting parity.
    
    # Let dx = |sx - tx| and dy = |sy - ty|.
    # The total cost is dy + cost_x.
    # For a fixed y, the cost to move dx is:
    # If we start at x and move to x + dx:
    # The boundaries are at x+1, x+2, ..., x+dx.
    # Boundary k is free if (k-1) + y is even.
    # Otherwise, it costs 1.
    
    # In any sequence of 2 horizontal steps, exactly one is free 
    # and one costs 1, regardless of y.
    # So for dx, the cost is dx // 2 + (1 if dx % 2 == 1 and (sx + sy) % 2 != 0 else 0).
    # Wait, let's re-evaluate:
    # Boundary k (between x+k-1 and x+k) is free if (x+k-1) + y is even.
    # If dx = 1: cost is 0 if (sx + sy) is even, else 1.
    # If dx = 2: cost is always 1 (one boundary is even, one is odd).
    # If dx = 3: cost is 1 if (sx + sy) is even, else 2.
    
    # General formula for cost_x:
    # If (sx + sy) % 2 == 0:
    #   dx=0: 0, dx=1: 0, dx=2: 1, dx=3: 1, dx=4: 2...
    #   This is dx // 2.
    # If (sx + sy) % 2 != 0:
    #   dx=0: 0, dx=1: 1, dx=2: 1, dx=3: 2, dx=4: 2...
    #   This is (dx + 1) // 2.
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    
    # We can choose to move horizontally at any y between sy and ty.
    # To minimize cost_x, we want to pick y such that (sx + y) % 2 == 0.
    # If sy == ty, we are stuck with sy.
    # If sy != ty, we can pick y = sy or y = sy + 1 to make (sx + y) even.
    # Thus, if dy > 0, we can always achieve the cost dx // 2.
    # If dy == 0, the cost is dx // 2 if (sx + sy) % 2 == 0 else (dx + 1) // 2.
    
    # However, the problem says we can move in any direction.
    # The most efficient way to move is to utilize the "free" X-boundaries.
    # If we move vertically, we always pay.
    # The only way to reduce X-cost is to change the parity of y.
    # But changing parity of y costs 1 toll.
    # If (sx + sy) % 2 != 0, we can pay 1 to move to sy + 1, 
    # then the X-cost becomes dx // 2.
    # Total cost = 1 + dx // 2.
    # Compare this to the cost if we stayed at sy: (dx + 1) // 2.
    # Note that (dx + 1) // 2 is always <= 1 + dx // 2.
    # So we only care about the parity of (sx + sy) if we never move vertically.
    # But we MUST move vertically if sy != ty.
    
    # Let's refine:
    # If sy == ty:
    #    cost = (dx + 1) // 2 if (sx + sy) % 2 != 0 else dx // 2
    # If sy != ty:
    #    We must pay dy. We can pick the parity of y that minimizes X-cost.
    #    The best X-cost is dx // 2 (by picking y such that sx + y is even).
    #    Since dy >= 1, we can always reach such a y.
    #    cost = dy + dx // 2.
    
    # Wait, if sy != ty, does the vertical movement "cost" the parity shift?
    # Yes, the vertical movement is the toll.
    # Example 1: S(5,0), T(2,5). dx=3, dy=5.
    # sx+sy = 5 (odd). 
    # If we move X first: cost is (3+1)//2 = 2. Then Y: 5. Total 7.
    # If we move Y first: cost is 5. Then X: 3//2 = 1. Total 6.
    # But the sample says 5. How?
    # "Move left 1 (0 toll), up 1 (1 toll), left 1 (0 toll), up 3 (3 tolls), left 1 (0 toll), up 1 (1 toll)."
    # Total toll: 0 + 1 + 0 + 3 + 0 + 1 = 5.
    # In this path, he moves X, then Y, then X, then Y...
    # He is utilizing the fact that at different Y levels, different X-boundaries are free.
    # At y=0, x=5 is the start of a tile (5+0=5 odd), so moving to x=4 costs 1.
    # Wait, the rule is: A_{i,j} and A_{i+1,j} are one tile if i+j is even.
    # For S(5,0), i=5, j=0. i+j=5 (odd). 
    # The tile is A_{5,0} and A_{6,0} (since 5+0 is odd, the tile is NOT these two).
    # Actually, if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # For i=4, j=0: 4+0=4 (even), so A_{4,0} and A_{5,0} are one tile.
    # Starting at (5.5, 0.5) means we are in A_{5,0}.
    # Moving to (4.5, 0.5) means entering A_{4,0}.
    # Since 4+0 is even, A_{4,0} and A_{5,0} are the same tile. Toll = 0.
    # Then moving to (4.5, 1.5) enters A_{4,1}. Toll = 1.
    # Now at A_{4,1}, i=4, j=1. i+j=5 (odd).
    # Moving to (3.5, 1.5) enters A_{3,1}. 
    # Is A_{3,1} and A_{4,1} the same tile? i=3, j=1 => i+j=4 (even). Yes!
    # So moving from x=4 to x=3 at y=1 costs 0.
    
    # This means for any step in X, we can choose to move to a Y where that 
    # specific X-boundary is free.
    # The cost to change Y is 1 per unit.
    # The cost to move X is 0 if we are at a Y that makes the boundary free.
    # For any X-boundary at x=k, it is free if k+y is even.
    # We can always find a Y in the range [sy, ty] that makes k+y even, 
    # UNLESS sy == ty and sx+sy is odd (for the first step).
    
    # Actually, the simplest observation:
    # We can move X for free if we can "toggle" the Y coordinate.
    # If dy > 0, we can move X at either Y or Y+1.
    # One of those will always have the boundary free.
    # So if dy > 0, the X-cost is effectively 0? 
    # No, because we can only move X for free if the boundary is the 
    # "even" one. But the tiles are 2x1. 
    # Every second X-boundary is ALWAYS a toll.
    # The boundaries are at x=k. The boundary between A_{k-1, j} and A_{k, j}
    # is free if (k-1)+j is even.
    # For a fixed j, the boundaries are:
    # k=1: (0+j) even?
    # k=2: (1+j) even?
    # One of these is always true, one is always false.
    # So for every 2 units of X, we must pay at least 1 toll.
    # The only way to avoid this is to change Y.
    # But changing Y also costs tolls.
    
    # Let's re-examine: 
    # To move from x to x+1, we pay 0 if x+y is even, else 1.
    # To move from y to y+1, we pay 1.
    # This is a shortest path problem on a graph.
    # The state is (x, y). But x and y are huge.
    # However, the cost depends on (x % 2, y % 2).
    # Let's use the property that we can move any distance n.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # min(
    #    dy + (dx // 2) + (1 if dx % 2 == 1 and (sx + sy + dy) % 2 != 0 else 0),
    #    ...
    # )
    # Actually, the most efficient way is to move X in blocks of 2.
    # Each block of 2 in X costs 1 toll (at the "odd" boundary).
    # We can choose whether to take the toll at the first or second boundary
    # by shifting Y by 1.
    # But shifting Y also costs 1.
    # The key is: we can move X for free if we are at the correct Y.
    # If we move Y, we pay.
    # The optimal strategy is to move X as much as possible for free,
    # then move Y, then move X for free again.
    # But we can only move X for free for 1 unit, then we hit a boundary 
    # that is only free if Y changes.
    # So: Move X (free), Move Y (1), Move X (free), Move Y (1)...
    # This looks like we pay 1 for every 1 unit of Y and 1 for every 2 units of X.
    # Total = dy + dx // 2 + (1 if dx % 2 == 1 and (sx + sy) % 2 != 0 else 0)
    # Wait, if we move Y first, the parity of (sx + sy) changes.
    # If sy != ty, we can choose to start moving X at sy or sy + 1.
    # If we start at sy, cost is (dx + 1)//2 if sx+sy is odd, else dx//2.
    # If we start at sy+1, cost is 1 + (dx + 1)//2 if sx+sy+1 is odd, else 1 + dx//2.
    # Since we must move dy anyway, we can just pick the best parity.
    # The best X-cost is always dx // 2, provided we can pick