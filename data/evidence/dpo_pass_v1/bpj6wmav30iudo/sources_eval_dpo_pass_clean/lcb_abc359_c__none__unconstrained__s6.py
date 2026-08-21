The constraint to avoid explicit loops and recursion forces a functional programming style, which encourages the use of declarative expressions over imperative state changes. For this problem, I will use basic arithmetic operations to calculate the Manhattan distance and coordinate transformations, as the minimum toll is derived from the sum of absolute differences of the coordinates after accounting for the tiling pattern.

```python
import sys

def solve():
    # Read input from standard input
    # Using map and split to handle the input without explicit loops
    input_data = sys.stdin.read().split()
    
    # Assign coordinates
    # Sx, Sy: Start point; Tx, Ty: Target point
    # The input is guaranteed to be four integers
    coords = list(map(int, input_data))
    Sx, Sy, Tx, Ty = coords[0], coords[1], coords[2], coords[3]

    # The tiling rule: A_{i,j} and A_{i+1,j} are the same tile if i+j is even.
    # This means tiles are 2x1 rectangles.
    # If i+j is even, the tile covers [i, i+2] x [j, j+1].
    # If i+j is odd, the tile covers [i, i+1] x [j, j+1] AND [i-1, i] x [j, j+1] 
    # (depending on the neighbor).
    # Essentially, in row j, the boundaries between tiles are at x = k 
    # where k has the same parity as j.
    
    # Let's transform the coordinates to a system where each unit move 
    # in the new system corresponds to crossing a tile boundary.
    # For a fixed j, moving from x to x+1 costs 0 if x+j is even, and 1 if x+j is odd.
    # Moving from j to j+1 always costs 1 because tiles are only 2x1 (horizontal).
    
    # The cost to move from (Sx, Sy) to (Tx, Ty):
    # 1. Vertical cost: |Sy - Ty|
    # 2. Horizontal cost: 
    #    In row j, the boundary is at x such that x % 2 == j % 2.
    #    The number of boundaries between Sx and Tx in row j is:
    #    floor(|Sx - Tx| / 2) + (1 if (|Sx - Tx| % 2 == 1 and (Sx + j) % 2 == 1) else 0)
    # However, we can optimize by choosing the row j (either Sy or Ty) 
    # that minimizes the horizontal cost.
    
    # Let dx = |Sx - Tx| and dy = |Sy - Ty|
    dx = abs(Sx - Tx)
    dy = abs(Sy - Ty)
    
    # The horizontal cost in row j is:
    # If dx is even, cost is dx // 2 regardless of j.
    # If dx is odd, cost is dx // 2, but we pay an extra 1 if (Sx + j) is odd.
    # We can choose to move horizontally at Sy or Ty.
    # The extra cost is 1 if (Sx + Sy) % 2 == 1 AND (Sx + Ty) % 2 == 1.
    # But wait, we can move to any row j between Sy and Ty.
    # If dy > 0, we can pick j such that (Sx + j) is even, making the extra cost 0.
    # If dy == 0, we are stuck with j = Sy.
    
    # Correct Logic:
    # Vertical distance dy always costs dy.
    # Horizontal distance dx:
    # Each 2 units of dx costs 1 toll.
    # If dx is odd, the remaining 1 unit costs 1 toll IF we cannot find a row j
    # in our path where (Sx + j) is even.
    # We can find such a j if dy > 0.
    # If dy == 0, we pay 1 extra if (Sx + Sy) % 2 == 1.
    
    # Let's re-evaluate:
    # The cost to move from (Sx, Sy) to (Tx, Ty) is:
    # dy + (dx // 2) + (1 if (dx % 2 == 1 and dy == 0 and (Sx + Sy) % 2 == 1) else 0)
    # Wait, Sample 1: (5,0) to (2,5). dx=3, dy=5. 
    # Cost = 5 + (3 // 2) + 0 = 5 + 1 = 6? Sample says 5.
    # Let's re-read: "Choose a direction... and a positive integer n. Move n units."
    # This means we can move diagonally by alternating.
    # Actually, the cost is simply:
    # (|Sx - Tx| + |Sy - Ty|) // 2 if we can optimize.
    # Let's use the property: the distance is (|Sx - Tx| + |Sy - Ty|) 
    # but we get a discount of 1 for every 2 units of horizontal movement 
    # if we are in a row that allows it.
    
    # The actual minimum toll is:
    # abs(Sy - Ty) + max(0, (abs(Sx - Tx) - (abs(Sy - Ty) % 2 == 0 and 0 or 1)) // 2)
    # No, the simplest form for this specific tiling is:
    # toll = abs(Sy - Ty) + (abs(Sx - Tx) + 1) // 2
    # But we can reduce toll by 1 if we move horizontally in a row where 
    # the start and end are in the same tile.
    
    # Let',s use the coordinate transformation:
    # A point (x, y) is in tile ( (x + (y % 2)) // 2, y )
    # Let X' = (x + (y % 2)) // 2, Y' = y
    # The distance is |X's - X't| + |Y's - Y't|
    # However, the "toll" is paid when ENTERING a tile.
    # Starting tile is free.
    
    # Let's use the logic from Sample 1: (5,0) to (2,5)
    # Sx=5, Sy=0 -> X's = (5 + 0)//2 = 2, Y's = 0
    # Tx=2, Ty=5 -> X't = (2 + 1)//2 = 1, Y't = 5
    # Distance = |2 - 1| + |0 - 5| = 1 + 5 = 6.
    # Sample 1 says 5. The starting tile is not paid.
    # So 6 - 1 = 5.
    
    # Sample 2: (3,1) to (4,1)
    # Sx=3, Sy=1 -> X's = (3 + 1)//2 = 2, Y's = 1
    # Tx=4, Ty=1 -> X't = (4 + 1)//2 = 2, Y't = 1
    # Distance = |2 - 2| + |1 - 1| = 0.
    # 0 - 1 is -1, but we take max(0, ...)
    
    # Formula: max(0, abs((Sx + (Sy % 2)) // 2 - (Tx + (Ty % 2)) // 2) + abs(Sy - Ty))
    # Wait, the starting tile is free, but we only subtract 1 if we actually moved.
    # If we are already in the target tile, cost is 0.
    
    # Let',s check Sample 3:
    # Sx = 2552608206527595, Sy = 5411232866732612
    # Tx = 771856005518028, Ty = 7206210729152763
    # X's = (2552608206527595 + 0) // 2 = 1276304103263797
    # Y's = 5411232866732612
    # X't = (771856005518028 + 1) // 2 = 385928002759014
    # Y't = 7206210729152763
    # Dist = |1276304103263797 - 385928002759014| + |5411232866732612 - 7206210729152763|
    # Dist = 890376100504783 + 1794977862420151 = 2685353962924934
    # This doesn't match Sample 3 (1794977862420151).
    
    # Re-evaluating: The cost is simply the Manhattan distance in the (X', Y') space
    # BUT we can move diagonally in (X, Y) to change parity.
    # The minimum toll is actually:
    # abs(Sy - Ty) + max(0, (abs(Sx - Tx) - (1 if (Sy % 2 != Ty % 2) else 0)) // 2)
    # No, that's not it.
    
    # Correct Logic:
    # To get from (Sx, Sy) to (Tx, Ty), you must cover vertical distance |Sy - Ty|.
    # Each vertical step costs 1.
    # For horizontal distance |Sx - Tx|, you can use the "free" half of the 2x1 tiles.
    # In any row j, you can move 2 units horizontally for the cost of 1.
    # If you move vertically, you can change the parity of j to make the 
    # horizontal move more efficient.
    
    # The minimum cost is:
    # abs(Sy - Ty) + (abs(Sx - Tx) + 1) // 2
    # But if we move vertically, we can potentially save 1 toll if the 
    # horizontal distance is odd and we can align the parity.
    # If |Sy - Ty| > 0, we can always pick the best parity for the horizontal move.
    # The cost is: abs(Sy - Ty) + abs(Sx - Tx) // 2
    # If |Sy - Ty| == 0, and abs(Sx - Tx) is odd, we check if (Sx + Sy) is even.
    # If (Sx + Sy) is even, the first step is free, so (dx+1)//2 - 1.
    # If (Sx + Sy) is odd, the first step costs 1, so (dx+1)//2.
    
    # Let's test Sample 1: 5, 0, 2, 5 -> dx=3, dy=5. Cost = 5 + 3//2 = 6. Still 6.
    # Wait, Sample 1 says 5. 5 + 0 = 5.
    # If dy > 0, we can move 1 unit vertically, then we are in a new row.
    # The vertical move costs 1. The horizontal move dx=3 costs (3//2) = 1.
    # Total = 5 + 1 = 6. Still not 5.
    
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Sample 1: (5, 0.5) to (2, 5.5)
    # Row 0: x=5 is in tile {(5,0),(6,0)} because 5+0 is odd? No, i+j even.
    # i=4, j=0: 4+0=4 (even) -> A_{4,0} and A_{5,0} are one tile.
    # So (5, 0.5) is in tile covering x in [4, 6].
    # Move left to x=4.5: still in same tile. Cost 0.
    # Move up to y=1.5: enter tile A_{4,1}. Cost 1.
    # Move left to x=3.5: A_{4,1} and A_{3,1} are one tile (3+1=4 even). Cost 0.
    # Move up to y=5.5: enter 4 tiles. Cost 4.
    # Total = 1 + 4 = 5.
    
    # This