The constraint to avoid explicit loops and recursion forces a functional programming style, which emphasizes immutable data transformations and declarative logic over imperative state changes. In Python, this encourages the use of high-level abstractions that describe *what* to compute rather than *how* to step through it. For this problem, I will use basic arithmetic operations and the `abs` function to calculate the Manhattan-like distance dictated by the tiling pattern, as the minimum toll is derived from the coordinate differences.

```python
import sys

def solve():
    # Read input from standard input
    # Using map and split to handle the input without explicit loops
    input_data = sys.stdin.read().split()
    
    # Assign coordinates based on the input list
    # S_x, S_y, T_x, T_y
    coords = list(map(int, input_data))
    
    # The tiling pattern is as follows:
    # If i + j is even, (i, j) and (i+1, j) are the same tile.
    # This means horizontal movement is "free" if we are in a tile 
    # that spans x and x+1.
    # Specifically, a tile is defined by (floor(x), floor(y))
    # If (i + j) is even, the tile is {(i, j), (i+1, j)}.
    # If (i + j) is odd, the tile is {(i, j), (i, j+1)}.
    # Wait, the rule says: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # This implies a brick-like pattern.
    # Let's analyze the cost to move from (Sx, Sy) to (Tx, Ty).
    # The cost is effectively the L1 distance, but we can move 
    # horizontally for free if we are in the "wide" part of the tile.
    # Actually, the problem simplifies to:
    # Each step in Y always costs 1.
    # A step in X costs 1 only if we cannot use the 2x1 horizontal tile.
    # However, since we can move any n units, we can move to any X 
    # and then move Y.
    # The optimal strategy is to move diagonally in a sense.
    # The cost is max(|Sx - Tx|, |Sy - Ty|) if we consider the grid 
    # as a graph where some edges are 0.
    # For this specific tiling:
    # The distance is |Sy - Ty| + max(0, |Sx - Tx| - |Sy - Ty|) 
    # if we align the parity correctly.
    # Actually, the minimum toll is simply (|Sx - Tx| + |Sy - Ty|) // 2 
    # if we consider the tiles as nodes in a graph.
    # But the correct formula for this specific tiling is:
    # Let dx = |Sx - Tx| and dy = |Sy - Ty|
    # The cost is (dx + dy) // 2 if we can leverage the 2x1 tiles.
    # Since we can move n units, we can move to any (x, y) 
    # such that we only pay for the "shorter" dimension's transitions.
    # The actual minimum toll is:
    # abs( (Sx + Sy) - (Tx + Ty) ) // 2 + abs( (Sx - Sy) - (Tx - Ty) ) // 2
    # No, that's for a different grid.
    # For this grid: the cost is max(|Sx - Tx|, |Sy - Ty|) 
    # if we can move diagonally. But we moveににに.
    # Let's re-evaluate: 
    # To move from (0,0) to (2,5):
    # (0,0) is in tile with (1,0). 
    # Move to (1,0) cost 0. Move to (1,1) cost 1.
    # (1,1) is in tile with (2,1). Move to (2,1) cost 0.
    # Move to (2,2) cost 1...
    # The cost is simply the L1 distance divided by 2, rounded up?
    # Sample 1: 5,0 to 2,5 -> dx=3, dy=5. L1=8. 8//2 = 4. Sample says 5.
    # Sample 2: 3,1 to 4,1 -> dx=1, dy=0. L1=1. 1//2 = 0. Sample says 0.
    # The pattern is: if i+j is even, (i,j) and (i+1,j) are one tile.
    # This means if we are at (i,j) and i+j is even, moving to i+1 is free.
    # If i+j is odd, moving to j+1 is free.
    # This is a Manhattan distance on a graph where edges alternate 0 and 1.
    # The distance is |Sx - Tx| + |Sy - Ty| - (number of free edges).
    # The maximum number of free edges we can take is ( |Sx-Tx| + |Sy-Ty| ) // 2.
    # So the cost is (|Sx-Tx| + |Sy-Ty|) - (|Sx-Tx| + |Sy-Ty|) // 2 = (|Sx-Tx| + |Sy-Ty| + 1) // 2.
    # Wait, Sample 1: (3+5+1)//2 = 4. Still not 5.
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    # This means for a fixed j, the x-intervals [0,1][1,2], [2,3][3,4]...
    # are paired if j is even. If j is odd, [0,1][0,1] is one tile? No.
    # If j is even: (0,j)-(1,j) same, (2,j)-(3,j) same...
    # If j is odd: (1,j)-(2,j) same, (3,j)-(4,j) same...
    # This is exactly the brick wall pattern.
    # The distance between (Sx, Sy) and (Tx, Ty) in this metric is:
    # |Sy - Ty| + (|Sx - Tx| + 1) // 2 if we are forced to move X.
    # Actually, the distance is |Sy - Ty| + max(0, |Sx - Tx| - (1 if parity matches else 0))
    # Let',s use the property: 
    # Cost = |Sy - Ty| + (|Sx - Tx| // 2) if parity of Sy and Sx allows.
    # Correct logic for brick wall:
    # The cost is |Sy - Ty| + (|Sx - Tx| + 1) // 2 if we start at an "odd" X for that Y.
    # Let's use the formula: 
    # dx = abs(Sx - Tx), dy = abs(Sy - Ty)
    # If we move only vertically, cost is dy. 
    # Then we need to cover dx. Each 2 units of dx cost 1.
    # Total = dy + (dx + 1) // 2.
    # Sample 1: 5 + (3+1)//2 = 5 + 2 = 7. Still not 5.
    # Sample 1 again: 5,0 to 2,5. dx=3, dy=5.
    # (5,0) is in tile with (4,0) because 4+0 is even.
    # (5,0) is NOT in tile with (6,0) because 5+0 is odd.
    # To move from x=5 to x=2:
    # x=5 -> x=4 (toll 0 if 4+0 even), x=4 -> x=3 (toll 1), x=3 -> x=2 (toll 0 if 2+0 even).
    # So x=5 to x=2 costs 1.
    # Then y=0 to y=5 costs 5. Total 6? Sample says 5.
    # Ah, we can move Y first.
    # y=0 -> y=5 costs 5. Then x=5 -> x=2.
    # At y=5, x=5: 5+5=10 (even), so (5,5) and (6,5) are same.
    # (4,5) and (5,5) are different. (3,5) and (4,5) are same.
    # x=5 -> x=4 (toll 1), x=4 -> x=3 (toll 0), x=3 -> x=2 (toll 1).
    # Total 5 + 2 = 7.
    # What if we move diagonally?
    # (5,0) -> (5,1) [toll 1] -> (4,1) [toll 0] -> (4,2) [toll 1] -> (3,2) [toll 0] -> (3,3) [toll 1] -> (2,3) [toll 0] -> (2,5) [toll 2]
    # Total: 1+1+1+2 = 5.
    # This looks like: for each step in Y, we can move 1 unit in X for free.
    # The cost is max(|Sy - Ty|, (|Sx - Tx| + |Sy - Ty| + 1) // 2).
    # Sample 1: max(5, (3+5+1)//2) = max(5, 4) = 5.
    # Sample 2: max(0, (1+0+1)//2) = max(0, 1) = 1. Sample 2 says 0.
    # Wait, Sample 2: (3,1) to (4,1). 3+1=4 (even). 
    # Rule: i+j even => A_{i,j} and A_{i+1,j} are same tile.
    # i=3, j=1 => 3+1=4 (even) => A_{3,1} and A_{4,1} are same tile.
    # So (3.5, 1.5) and (4.5, 1.5) are in the same tile. Toll = 0.
    # My formula max(|Sy-Ty|, ...) gives 1.
    # The actual logic:
    # Let dx = |Sx - Tx|, dy = |Sy - Ty|.
    # We can move 1 unit of dx for free for every 1 unit of dy, 
    # PROVIDED we are at the correct parity.
    # Since we can choose to move Y first or X first, 
    # and we can move any n, we can always align parity.
    # The only constraint is that we can't move more than 1 X for free per 1 Y.
    # Actually, the cost is simply:
    # If we move from (Sx, Sy) to (Tx, Ty), we can use the "free" X-move 
    # whenever (i+j) is even.
    # This means in any 2x2 block, there is one free horizontal move 
    # and one free vertical move? No.
    # Let',s use the property: 
    # The distance is (|Sx - Tx| + |Sy - Ty|) // 2, 
    # but we must account for the starting and ending tile boundaries.
    # Let's use the coordinate transformation:
    # The tiles are 2x1. This is like a Manhattan distance on a 
    # grid where some edges are 0.
    # The minimum toll is (|Sx - Tx| + |Sy - Ty|) // 2 
    # if we can start and end on the "free" edges.
    # After analyzing the brick pattern, the distance is:
    # abs( (Sx + Sy) - (Tx + Ty) ) // 2 + abs( (Sx - Sy) - (Tx - Ty) ) // 2
    # No, that's for a 45-degree rotated grid.
    # Let's try: (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2 
    # but Sample 2 (3,1 to 4,1) is 0.
    # In Sample 2, Sx=3, Tx=4, Sy=1, Ty=1. dx=1, dy=0.
    # Since 3+1=4 is even, A_{3,1} and A_{4,1} are the same tile.
    # So if dx=1 and dy=0, and (Sx+Sy) is even, cost is 0.
    # If dx=1 and dy=0, and (Sx+Sy) is odd, cost is 1.
    # This means the "free" move is available at (i,j) if i+j is even.
    # This is equivalent to saying we can move from (i,j) to (i+1