```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, the boundaries between tiles are at x = k where k+j is odd.
    # Let's transform the coordinates to a space where we can use Manhattan distance.
    # A common trick for "brick" patterns is to map (x, y) to a coordinate system
    # that linearizes the cost.
    # Let u = x + y and v = x - y.
    # However, a simpler approach for this specific tiling:
    # The cost to move between (sx, sy) and (tx, ty) in this grid is:
    # max(|sx - tx|, |sy - ty|) is not quite right because of the offsets.
    # The correct distance for this specific tiling is:
    # cost = max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # Wait, that's for a different tiling. Let's re-evaluate.
    
    # Let's use the property:
    # To move from (sx, sy) to (tx, ty):
    # The vertical distance is |sy - ty|. Each vertical step always enters a new tile.
    # The horizontal distance is |sx - tx|. 
    # Because of the offset, moving horizontally might be "free" if you are in the 
    # same 2x1 tile.
    
    # The distance is actually:
    # Let dx = tx - sx, dy = ty - sy.
    # The cost is max(|dx + dy|, |dx - dy|) // 2 
    # But we must adjust for the parity of the starting cell.
    # Let's use the coordinate transformation:
    # X = x + (y % 2), Y = y
    # This doesn't quite work.
    
    # Correct logic for this specific tiling:
    # The distance is max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # But we need to account for the starting parity.
    # If we shift the coordinates such that the "even" rule is normalized:
    # Let x' = sx, y' = sy.
    # The distance is max(abs(sx + sy - (tx + ty)), abs(sx - sy - (tx - ty))) // 2
    # Let's check Sample 1: 5 0 to 2 5
    # |(5+0) - (2+5)| = |5 - 7| = 2
    # |(5-0) - (2-5)| = |5 - (-3)| = 8
    # max(2, 8) // 2 = 4. Sample 1 says 5.
    
    # Let's try: cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # This is also not quite right.
    
    # The actual distance for this tiling is:
    # Let x1, y1 be start and x2, y2 be end.
    # The cost is max(|x1 - x2 + (y1 % 2) - (y2 % 2)|, |y1 - y2|) 
    # No, the simplest general form for this problem is:
    # cost = max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # But we must handle the parity of the starting cell relative to the tile boundaries.
    # If (sx + sy) is even, the tile is { (sx, sy), (sx+1, sy) }.
    # If (sx + sy) is odd, the tile is { (sx-1, sy), (sx, sy) }.
    
    # Let's normalize sx so that the tile always covers {sx, sx+1}.
    # If (sx + sy) is odd, we can treat the position as (sx-1, sy) for the purpose of 
    # the "even" rule, but we must be careful.
    
    # Correct formula:
    # Let X = sx + (1 if (sx + sy) % 2 != 0 else 0)
    # Let Y = sy
    # Let X2 = tx + (1 if (tx + ty) % 2 != 0 else 0)
    # Let Y2 = ty
    # This is still guessing. Let's use the property:
    # The distance is max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # Adjusted for the "half-steps":
    # The distance is (max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) + 1) // 2
    # Wait, Sample 1: max(2, 8) = 8. (8+1)//2 = 4. Still 4.
    
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # This means in row j, tiles are [0,1][2,3]... if j is even, and [1,2][3,4]... if j is odd.
    # This is exactly the coordinate system of a chessboard where tiles are dominoes.
    # The distance is:
    # dist = max(|sx - tx|, |sy - ty|) if we could move diagonally.
    # But we can only move H/V.
    # The cost is:
    # Let x' = sx, y' = sy.
    # If (x' + y') is odd, we are in the second half of a tile.
    # The cost is max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # Let's try: cost = (max(abs(sx + sy - (tx + ty)), abs(sx - sy - (tx - ty))) + 1) // 2
    # Still 4. What is missing?
    # The parity of the start/end points.
    # Let's use: 
    # x_norm = sx + (1 if (sx + sy) % 2 != 0 else 0)
    # y_norm = sy
    # x2_norm = tx + (1 if (tx + ty) % 2 != 0 else 0)
    # y2_norm = ty
    # This is not working.
    
    # Final attempt at the logic:
    # The distance is max(|sx - tx|, |sy - ty|) if we can move diagonally.
    # In this specific tiling, the distance is:
    # abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Sample 1: abs(0 - 5) + max(0, (abs(5 - 2) - 5 + 1) // 2) = 5 + max(0, -1 // 2) = 5 + 0 = 5.
    # Sample 2: abs(1 - 1) + max(0, (abs(3 - 4) - 0 + 1) // 2) = 0 + max(0, 2 // 2) = 1.
    # Wait, Sample 2 output is 0. My formula gives 1.
    # In Sample 2: (3,1) and (4,1). i=3, j=1. i+j = 4 (even).
    # Rule: A_{i,j} and A_{i+1,j} are same tile.
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    
    # The correct logic:
    # A point (x, y) belongs to tile ( (x + (y%2)) // 2, y )
    # Let X = (sx + (sy % 2)) // 2, Y = sy
    # Let X2 = (tx + (ty % 2)) // 2, Y2 = ty
    # The distance between (X, Y) and (X2, Y2) in a grid where you can move 
    # to 8 neighbors (including diagonals) is max(|X-X2|, |Y-Y2|).
    # But we can only move H/V. 
    # Actually, the movement rules allow us to change X or Y.
    # Moving Y by 1 costs 1. Moving X by 1 costs 1.
    # But moving X and Y simultaneously (diagonally) is not allowed.
    # However, we can move X, then Y, then X...
    # The cost is simply the Manhattan distance in the transformed space?
    # Let's check Sample 1:
    # S: (5, 0) -> X = (5 + 0)//2 = 2, Y = 0
    # T: (2, 5) -> X2 = (2 + (5%2))//2 = (2+1)//2 = 1, Y2 = 5
    # Manhattan: |2-1| + |0-5| = 1 + 5 = 6. Still not 5.
    # Chebyshev: max(|2-1|, |0-5|) = 5. Matches Sample 1!
    # Sample 2:
    # S: (3, 1) -> X = (3 + 1)//2 = 2, Y = 1
    # T: (4, 1) -> X2 = (4 + 1)//2 = 2, Y2 = 1
    # Chebyshev: max(|2-2|, |1-1|) = 0. Matches Sample 2!
    # Sample 3:
    # S: 2552608206527595, 5411232866732612
    # T: 771856005518028, 7206210729152763
    # X = (2552608206527595 + 0) // 2 = 1276304103263797
    # Y = 5411232866732612
    # X2 = (771856005518028 + 1) // 2 = 385928002759014
    # Y2 = 7206210729152763
    # max(|1276304103263797 - 385928002759014|, |5411232866732612 - 7206210729152763|)
    # max(890376100504783, 1794977862420151) = 1794977862420151. Matches Sample 3!

    # Final Formula:
    # X = (sx + (sy % 2)) // 2
    # Y = sy
    # X2 = (tx + (ty % 2)) // 2
    # Y2 = ty
    # result = max(abs(X - X2), abs(Y - Y2))
    
    # Using reduce to perform the calculation without explicit assignments
    res = reduce(lambda acc, val: val, [
        (
            (sx + (sy % 2)) // 2,
            sy,
            (tx + (ty % 2)) // 2,
            ty
        ),
        lambda coords: max(abs(coords[0] - coords[2]), abs(coords[1] - coords[3]))
    ])
    # The reduce above is slightly wrong in structure, let's just use a list comprehension and max.
    
    # Corrected reduce/map structure:
    ans = (lambda s_x, s_y, t_x, t_y: 
           max(abs(((s_x + (s_y % 2)) // 2) - ((t_x + (t_y % 2)) // 2)), 
               abs(s_y