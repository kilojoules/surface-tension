The constraint to avoid explicit loops and recursion forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state updates. In Python, this encourages the use of high-level abstractions that describe *what* to compute rather than *how* to step through it. For this specific problem, since the solution relies on a closed-form mathematical formula derived from the Manhattan distance and the parity of the grid coordinates, I will use basic arithmetic operations and the `abs()` function within a `print()` statement to achieve the result in a single expression.

```python
import sys

def solve():
    # Read all input values into a list
    # map(int, sys.stdin.read().split()) handles the input without explicit loops
    input_data = list(map(int, sys.stdin.read().split()))
    
    # Assign coordinates based on the input list
    # Sx, Sy = input_data[0], input_data[1]
    # Tx, Ty = input_data[2], input_data[3]
    
    # The tiling pattern is: if (i + j) is even, (i, j) and (i+1, j) are one tile.
    # This means tiles are 2x1 blocks aligned horizontally when i+j is even.
    # Let's transform the coordinates to a space where each move costs 1.
    # The cost to move from (Sx, Sy) to (Tx, Ty) in this specific grid 
    # is related to the Manhattan distance, but the "free" moves 
    # occur on the 2-unit wide tiles.
    # Specifically, the cost is (|Sx - Tx| + |Sy - Ty|) / 2, 
    # adjusted for the parity of the starting and ending cells.
    # Let x' = x, y' = y. A move in y always costs 1. 
    # A move in x costs 1 only if it crosses a tile boundary.
    # The boundary between (i, j) and (i+1, j) exists if i+j is odd.
    
    # The minimum toll is given by the formula:
    # (|Sx - Tx| + |Sy - Ty|) // 2 if the parity of (Sx+Sy) and (Tx+Ty) 
    # allows for optimal traversal, but more simply:
    # The distance is (|Sx - Tx| + |Sy - Ty|) / 2, rounded up 
    # depending on the relative positions.
    # After analyzing the grid: 
    # The cost is (|Sx - Tx| + |Sy - Ty|) // 2 if we consider 
    # the diagonal-like nature of the tiles.
    # Actually, the cost is simply (|Sx - Tx| + |Sy - Ty|) // 2 
    # if we align the coordinates.
    # Let's use the derived formula: 
    # cost = (abs(Sx - Tx) + abs(Sy - Ty)) // 2 
    # but we must account for the specific tiling offset.
    # The correct distance is:
    # Let dx = abs(Sx - Tx), dy = abs(Sy - Ty)
    # If we are in a tile that covers (i, j) and (i+1, j), 
    # we can move x by 1 for free.
    # This happens every other column.
    # The minimum toll is (abs(Sx - Tx) + abs(Sy - Ty)) // 2 
    # if we optimize the path.
    # However, the exact formula considering the (i+j) even rule is:
    # toll = (abs(Sx - Tx) + abs(Sy - Ty)) // 2 
    # but we must check if the parity of (Sx+Sy) and (Tx+Ty) 
    # requires an extra step.
    # Let's refine: 
    # Each step in Y costs 1. Each step in X costs 1 only if i+j is odd.
    # This is equivalent to a coordinate transformation:
    # Let u = x + y, v = x - y.
    # The distance is max(|u1 - u2|, |v1 - v2|) / 2 ? No.
    # The actual minimum toll is:
    # (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2 if (Sx+Sy)%2 != (Tx+Ty)%2 else ...
    # Actually, the simplest form is:
    # toll = (abs(Sx - Tx) + abs(Sy - Ty)) // 2
    # But we must account for the starting tile.
    # If Sx+Sy is even, we are at the left side of a 2x1 tile.
    # If Sx+Sy is odd, we are at the right side.
    # The distance is (|Sx - Tx| + |Sy - Ty|) / 2, 
    # rounded up if the parity of the manhattan distance 
    # doesn't match the grid's "free" directions.
    # Correct Formula:
    # Let dx = abs(Sx - Tx), dy = abs(Sy - Ty)
    # The cost is (dx + dy) // 2, but if (dx + dy) is odd, 
    # we need to check if the "extra" 0.5 is required.
    # For this specific tiling:
    # Toll = (abs(Sx - Tx) + abs(Sy - Ty)) // 2 
    # if (Sx + Sy) % 2 == (Tx + Ty) % 2 
    # else (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2
    # This simplifies to: (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2 
    # ONLY IF the parity of the sum changes.
    # Wait, the Sample 1: (5,0) to (2,5). dx=3, dy=5. (3+5)//2 = 4. 
    # But output is 5. 
    # Sample 2: (3,1) to (4,1). dx=1, dy=0. (1+0)//2 = 0. Output 0.
    # Sample 1: Sx+Sy = 5 (odd), Tx+Ty = 7 (odd). 
    # Sample 2: Sx+Sy = 4 (even), Tx+Ty = 5 (odd).
    # Let's re-evaluate:
    # In Sample 1: 5+0 is odd. A_{5,0} is the right half of a tile.
    # To move to 2,5: dx=3, dy=5. 
    # The cost is actually: 
    # If we move Y, cost is 1. If we move X, cost is 1 only if we cross 
    # from i to i+1 where i+j is odd.
    # This is a shortest path on a graph.
    # The cost is (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2 
    # if we are "unlucky" with parity.
    # Let's use the property: 
    # Cost = abs((Sx + Sy) - (Tx + Ty)) // 2 + abs((Sx - Sy) - (Tx - Ty)) // 2
    # Sample 1: abs(5 - 7)//2 + abs(5 - (-3))//2 = 2//2 + 8//2 = 1 + 4 = 5.
    # Sample 2: abs(4 - 5)//2 + abs(2 - 3)//2 = 1//2 + 1//2 = 0 + 0 = 0.
    # Sample 3: Sx=2552608206527595, Sy=5411232866732612, Tx=771856005518028, Ty=7206210729152763
    # Sx+Sy = 7963841073260207, Tx+Ty = 8000000000000000? No.
    # Let', use the formula: 
    # result = (abs((Sx + Sy) - (Tx + Ty)) + 1) // 2 + (abs((Sx - Sy) - (Tx - Ty)) + 1) // 2
    # No, that's not it.
    # The correct formula for this specific grid is:
    # cost = (abs(Sx - Tx) + abs(Sy - Ty)) // 2 
    # but we must account for the "staggered" nature.
    # The distance is actually:
    # (| (Sx+Sy) - (Tx+Ty) | + | (Sx-Sy) - (Tx-Ty) |) // 2
    # Sample 1: (| 5 - 7 | + | 5 - (-3) |) // 2 = (2 + 8) // 2 = 5.
    # Sample 2: (| 4 - 5 | + | 2 - 3 |) // 2 = (1 + 1) // 2 = 1. 
    # Wait, Sample 2 output is 0.
    # Let's try: (abs(Sx - Tx) + abs(Sy - Ty)) // 2 
    # Sample 1: (3 + 5) // 2 = 4. (Wrong)
    # Sample 2: (1 + 0) // 2 = 0. (Correct)
    # Let', reconsider: 
    # A move in Y always costs 1. A move in X costs 1 if i+j is odd.
    # This means moving (x, y) -> (x+1, y) costs 1 if x+y is odd, 0 if x+y is even.
    # This is exactly the distance in a grid where edges have weights.
    # The min cost is abs(Sy - Ty) + (abs(Sx - Tx) + (1 if (Sx+Sy)%2 != (Tx+Ty)%2 else 0)) // 2
    # Sample 1: 5 + (3 + (0)) // 2 = 5 + 1 = 6. (Wrong)
    # Let's try: The cost is abs(Sy - Ty) + max(0, (abs(Sx - Tx) - (1 if (Sx+Sy)%2 == 0 else 0)) // 2)
    # Sample 1: 5 + max(0, (3 - 0)//2) = 5 + 1 = 6. (Wrong)
    # Correct logic: 
    # To move from Sx to Tx, we need abs(Sx - Tx) steps.
    # Some are free. A step from x to x+1 is free if x+y is even.
    # For a fixed y, the x-steps are free, paid, free, paid...
    # The number of paid x-steps is (abs(Sx - Tx) + 1) // 2 if we start at an odd x+y.
    # But we can change y to change the parity!
    # If we move y by 1, we pay 1, but we might save 1 in x.
    # This is a shortest path on a graph where nodes are (x, y).
    # The cost is min(abs(Sy - Ty) + cost_x(Sx, Tx, Sy), 
    #                abs(Sy - Ty - 1) + cost_x(Sx, Tx, Sy + 1), ...)
    # Actually, the cost is simply:
    # abs(Sy - Ty) + (abs(Sx - Tx) // 2) 
    # if (Sx + Sy) is even and we moveに the right direction.
    # Let's use the most reliable formula for this problem:
    # cost = abs(Sy - Ty) + (abs(Sx - Tx) + (1 if (Sx + Sy) % 2 != 0 and (Tx + Ty) % 2 != 0 and abs(Sx-Tx)%2 != 0 else 0)) // 2
    # No. Let's use:
    # The cost is abs(Sy - Ty) + (abs(Sx - Tx) // 2) 
    # if (Sx + Sy) % 2 == 0 else (abs(Sx - Tx) + 1) // 2
    # Sample 1: 5 + (3 // 2) = 6 or (3+1)//2 = 2. 5+2=7.
    # Let' same look at Sample 1 again. 5,0 to 2,5.
    # (5,0) is odd. (4,0) is even. Move 5->4 costs 0.
    # (4,0) is even. Move 0->1 (Y) costs 1.
    # (4,1) is odd. Move 4->3 costs 0.
    # (3,1) is even. Move 1->4 (Y) costs 3.
    # (3,4) is odd. Move 3->2 costs 0.
    # (2,4) is even.