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
    # This means for a fixed j:
    # If j is even, tiles are [0,1], [2,3], [4,5]... (i=0, 2, 4...)
    # If j is odd, tiles are [-1,0], [1,2], [3,4]... (i=-1, 1, 3...)
    
    # Let's define a coordinate transformation to a space where 
    # the tiles are standard 2x1 blocks aligned to the axes.
    # In the original space, the "tile boundaries" for i are:
    # For even j: x = 0, 2, 4...
    # For odd j: x = 1, 3, 5...
    # This is equivalent to saying a boundary exists at x if (x + j) is even.
    
    # We want to find the minimum toll. 
    # Moving horizontally: we pay if we cross a boundary.
    # Moving vertically: we always pay because every A_{i,j} and A_{i,j+1} 
    # are in different tiles (since the rule only merges A_{i,j} and A_{i+1,j}).
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # 1. The vertical distance |sy - ty|.
    # 2. The horizontal cost. 
    # If we move horizontally at height j, we pay for every x such that (x+j) is even.
    # However, we can pick ANY height j to move horizontally.
    # To minimize horizontal cost, we want to move at a height j where the 
    # number of boundaries between sx and tx is minimized.
    
    # For a fixed j, the boundaries are x where x % 2 == j % 2.
    # The number of such x in (min(sx, tx), max(sx, tx)) is:
    # Let L = min(sx, tx), R = max(sx, tx).
    # We count x in {L+1, ..., R} such that x % 2 == j % 2.
    # This count is either floor((R-L)/2) or ceil((R-L)/2).
    # We can choose j to be even or odd to pick the minimum.
    
    # Actually, the problem is simpler:
    # We can move vertically to a height j, then move horizontally, then vertically to ty.
    # But we can also move horizontally at different heights.
    # The optimal strategy is to move to a height j that minimizes the horizontal cost,
    # then move horizontally, then move to ty.
    # The total cost is |sy - j| + |ty - j| + horizontal_cost(j).
    # Note that |sy - j| + |ty - j| is minimized for any j between sy and ty, 
    # and its minimum value is |sy - ty|.
    
    # So we just need to find j in [min(sy, ty), max(sy, ty)] that minimizes 
    # the number of x in (min(sx, tx), max(sx, tx)) such that x % 2 == j % 2.
    # If sy == ty, we can pick j = sy.
    # If sy != ty, we can pick j to be either even or odd (since the range contains both).
    
    # Let dx = abs(sx - tx).
    # The number of boundaries x in (L, R] is dx.
    # Half of these are even, half are odd.
    # The number of boundaries of a specific parity is dx // 2.
    # If dx is even, both parities have dx // 2 boundaries.
    # If dx is odd, one parity has dx // 2 and the other has (dx // 2) + 1.
    
    # Therefore, the minimum horizontal cost is always dx // 2.
    # The total cost is abs(sy - ty) + (abs(sx - tx) // 2).
    
    # Wait, let's check Sample 1: 5 0 to 2 5.
    # sx=5, sy=0, tx=2, ty=5.
    # abs(5-2)//2 + abs(0-5) = 3//2 + 5 = 1 + 5 = 6.
    # But the sample output says 5. Let's re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Sample 1: (5.5, 0.5) to (2.5, 5.5).
    # Move left 1: (4.5, 0.5). A_{5,0} and A_{4,0} are in the same tile (4+0 is even).
    # Toll: 0.
    # Move up 1: (4.5, 1.5). A_{4,0} and A_{4,1} are different tiles.
    # Toll: 1.
    # Move left 1: (3.5, 1.5). A_{4,1} and A_{3,1} are in the same tile (3+1 is even).
    # Toll: 0.
    # Move up 3: (3.5, 4.5). A_{3,1}, A_{3,2}, A_{3,3}, A_{3,4}.
    # A_{3,1} is the current tile. We enter A_{3,2}, A_{3,3}, A_{3,4}.
    # Toll: 3.
    # Move left 1: (2.5, 4.5). A_{3,4} and A_{2,4} are in the same tile (2+4 is even).
    # Toll: 0.
    # Move up 1: (2.5, 5.5). A_{2,4} and A_{2,5} are different tiles.
    # Toll: 1.
    # Total: 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # Analysis:
    # Horizontal move at height j from x1 to x2:
    # We pay if we cross a boundary. A boundary exists at x if (x+j) is even.
    # This is the same as x % 2 == j % 2.
    # The number of such x in (min(x1, x2), max(x1, x2)) is the cost.
    # Let L = min(x1, x2), R = max(x1, x2).
    # Count x in {L+1, ..., R} such that x % 2 == j % 2.
    # This count is (R - L) // 2 if (R-L) is even.
    # If (R-L) is odd, it's either (R-L)//2 or (R-L)//2 + 1.
    # We can choose j to make it (R-L)//2.
    
    # Total cost = abs(sy - ty) + (abs(sx - tx) // 2).
    # Let's check Sample 1 again: abs(0-5) + (abs(5-2)//2) = 5 + 3//2 = 5 + 1 = 6.
    # Still 6. Why?
    # In the sample path:
    # 1. Left 1: (5,0) -> (4,0). x=4. j=0. 4%2 == 0%2 is True. 
    # Wait, the rule is: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # For j=0, i=4: 4+0=4 (even). So A_{4,0} and A_{5,0} are one tile.
    # Moving from x=5.5 to x=4.5 means we are in A_{5,0} then A_{4,0}.
    # Since they are the same tile, toll is 0.
    # This matches my "boundary" logic: a boundary exists at x if (x-1)+j is odd.
    # No, the rule is: A_{i,j} and A_{i+1,j} are the same if i+j is even.
    # The boundary between A_{i,j} and A_{i+1,j} is the line x = i+1.
    # This boundary is "removed" if i+j is even.
    # So we pay for the boundary x = i+1 if i+j is odd.
    # i+1 = x  => i = x-1.
    # Boundary x is paid if (x-1)+j is odd, which means x+j is even.
    # This is exactly what I had: x % 2 == j % 2.
    
    # Let's re-calculate Sample 1 with the logic:
    # sx=5, tx=2, sy=0, ty=5.
    # Vertical distance = 5.
    # Horizontal distance = 3.
    # We can pick j=0, 1, 2, 3, 4, or 5.
    # For j=0: x in {3, 4, 5}. x+0 even? x=4. (1 boundary)
    # For j=1: x in {3, 4, 5}. x+1 even? x=3, 5. (2 boundaries)
    # Min horizontal cost is 1.
    # Total = 5 + 1 = 6.
    # BUT the sample says 5. Let me re-read again.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Path:
    # (5.5, 0.5) -> (4.5, 0.5): A_{5,0} to A_{4,0}. i=4, j=0. i+j=4 (even). Same tile. Cost 0.
    # (4.5, 0.5) -> (4.5, 1.5): A_{4,0} to A_{4,1}. Different tiles. Cost 1.
    # (4.5, 1.5) -> (3.5, 1.5): A_{4,1} to A_{3,1}. i=3, j=1. i+j=4 (even). Same tile. Cost 0.
    # (3.5, 1.5) -> (3.5, 4.5): A_{3,1} to A_{3,2}, A_{3,3}, A_{3,4}. Cost 3.
    # (3.5, 4.5) -> (2.5, 4.5): A_{3,4} to A_{2,4}. i=2, j=4. i+j=6 (even). Same tile. Cost 0.
    # (2.5, 4.5) -> (2.5, 5.5): A_{2,4} to A_{2,5}. Different tiles. Cost 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # Wait! The vertical moves also depend on the tiles.
    # A_{i,j} and A_{i,j+1} are ALWAYS different tiles.
    # But a horizontal move can be "free" if we are at the right height.
    # The total cost is:
    # (Number of vertical boundaries crossed) + (Number of horizontal boundaries crossed)
    # Vertical boundaries: always |sy - ty|.
    # Horizontal boundaries: we can pick ANY height j to move.
    # But we can only move horizontally at height j if we have already reached height j.
    # Actually, we can move horizontally at any height j between sy and ty.
    # To minimize the toll, we should pick the height j in [min(sy, ty), max(sy, ty)]
    # that has the minimum number of boundaries between sx and tx.
    # The number of boundaries at height j is the count of x in (min(sx, tx), max(sx, tx))
    # such that x+j is even.
    # This count is either dx // 2 or (dx + 1) // 2.
    # We can always pick j to be the one that gives dx // 2, PROVIDED that
    # the range [min(sy, ty), max(sy, ty)] contains both an even and an odd integer.
    # If sy