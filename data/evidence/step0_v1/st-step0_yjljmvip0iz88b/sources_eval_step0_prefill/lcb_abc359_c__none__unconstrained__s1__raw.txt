```python
import sys

def solve():
    # Read input from standard input
    try:
        line = sys.stdin.read().split()
        if not line:
            return
        sx, sy, tx, ty = map(int, line)
    except ValueError:
        return

    # The problem describes a tiling of the plane with 2x1 tiles.
    # Rule: If i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed row j:
    # If j is even: tiles are [0,1]U[1,2], [2,3]U[3,4]... (horizontal tiles starting at even i)
    # If j is odd: tiles are [-1,0]U[0,1], [1,2]U[2,3]... (horizontal tiles starting at odd i)
    # Wait, the rule says: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # Let's re-evaluate:
    # For a fixed j:
    # If j is even: i=0 (0+j even) -> A_{0,j} and A_{1,j} same tile. i=2 (2+j even) -> A_{2,j} and A_{3,j} same tile.
    # If j is odd: i=1 (1+j even) -> A_{1,j} and A_{2,j} same tile. i=3 (3+j even) -> A_{3,j} and A_{4,j} same tile.
    # This is a brick-wall pattern.
    
    # Let's define the tile ID for a cell (i, j).
    # In row j, the tiles are pairs (i, i+1) where i+j is even.
    # A cell (i, j) belongs to tile:
    # If (i+j) is even, it's the left half of the tile.
    # If (i+j) is odd, it's the right half of the tile.
    # Specifically, the tile is identified by ( (i if (i+j)%2==0 else i-1), j ).
    
    # The cost to move:
    # 1. Moving horizontally within a tile costs 0.
    # 2. Moving to a different tile costs 1.
    # However, the problem says "Each time he enters a tile, he pays a toll of 1."
    # This implies if he is already in tile T and moves to tile T', he pays 1.
    # If he moves within tile T, he pays 0.
    # The starting tile is free (he is already there).
    
    # Let's analyze the distance.
    # To move from (sx, sy) to (tx, ty):
    # The vertical distance is |sy - ty|. Each vertical step always enters a new tile.
    # The horizontal distance is |sx - tx|.
    # In a brick pattern, you can move horizontally for "free" if you are in the same tile.
    # Each row has tiles of length 2.
    # To move horizontally across the plane, you can leverage the offset of the bricks.
    # The optimal strategy to move from (sx, sy) to (tx, ty) is:
    # The cost is max(|sy - ty|, (|sx - tx| + 1) // 2) is NOT correct because of the stagger.
    
    # Correct logic for brick-wall distance:
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # If we move vertically dy steps, we can cover some horizontal distance.
    # In each row, we can move 1 unit horizontally for free (within the tile).
    # Because the tiles are staggered, moving 1 unit vertically and 1 unit horizontally
    # can sometimes be done by entering only one new tile.
    
    # Let's use the coordinate transformation for brick walls:
    # The distance is max(dy, (dx + dy + 1) // 2) ? No.
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # The minimum cost is max(dy, (dx + 1) // 2) if we can move diagonally.
    # But we can only move L, R, U, D.
    # A move U/D always costs 1. A move L/R costs 1 unless we stay in the same tile.
    # In any row, we can move 1 unit for free.
    # To move dx units horizontally, we need at least ceil(dx/2) tiles.
    # However, we can combine vertical and horizontal moves.
    # The actual distance in this specific brick layout is:
    # cost = max(dy, (dx + 1) // 2) is for a different problem.
    # For this problem:
    # The cost is dy + max(0, (dx - (dy + 1)) // 2 + 1) if dx > dy + 1 else dy.
    # Wait, let's simplify:
    # Each vertical step allows us to "shift" our horizontal position relative to the tile boundaries.
    # The minimum cost is max(dy, (dx + 1) // 2) is actually correct for the L1-like distance on bricks.
    # Let's test Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+1)//2) = 1? No, Sample 2 output is 0.
    # Why 0? (3,1) and (4,1). i=3, j=1. i+j = 4 (even). 
    # Rule: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # So A_{3,1} and A_{4,1} are in the same tile. Cost 0.
    
    # Let's refine:
    # Two cells (sx, sy) and (tx, ty) are in the same tile if sy == ty and (sx+sy)%2 == 0 and tx == sx+1
    # or (tx+sy)%2 == 0 and sx == tx+1.
    
    # The general formula for distance between (sx, sy) and (tx, ty) in this grid:
    # The cost is max(dy, (dx + 1) // 2) is almost right, but we must account for the 
    # specific tile alignment.
    # Let's use the property: cost = max(dy, (dx + 1) // 2) if we can move diagonally.
    # But we can't. However, we can move 1 unit horizontally and 1 unit vertically 
    # by entering only one new tile.
    # Example: (0,0) -> (1,0) is free. (1,0) -> (1,1) costs 1. (1,1) -> (2,1) is free.
    # So (0,0) to (2,1) costs 1. dx=2, dy=1. max(1, (2+1)//2) = 1.
    # (0,0) to (3,1): (0,0)->(1,0) [0], (1,0)->(1,1) [1], (1,1)->(2,1) [0], (2,1)->(3,1) [1]. Total 2.
    # dx=3, dy=1. max(1, (3+1)//2) = 2.
    
    # There is one edge case: if sx and tx are in the same tile and sy == ty, cost is 0.
    # Otherwise, the formula max(dy, (dx + 1) // 2) might be slightly off based on 
    # whether the start and end points are the "left" or "right" parts of their tiles.
    
    # Let's use the coordinate transformation:
    # A tile is identified by (X, Y) where X = (i if i+j is even else i-1) and Y = j.
    # The distance between (X1, Y1) and (X2, Y2) is:
    # cost = max(|Y1 - Y2|, (|X1 - X2| + 1) // 2) ? No.
    # Let's use the property that this is equivalent to a graph.
    # The distance is max(|sy - ty|, (|sx - tx| + 1) // 2) is for a different grid.
    # For this grid, the distance is:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # If dy == 0:
    #   If they are in the same tile, 0. Else, (dx + 1) // 2.
    #   Wait, if sy == ty, and they are not in the same tile, 
    #   you MUST move to another row and back to use the "free" horizontal moves.
    #   But moving to another row and back costs 2.
    #   Actually, if sy == ty, you can just move horizontally. 
    #   Each tile is 2 units wide. To move dx, you cross dx // 2 tiles.
    #   If they are in the same tile, 0.
    #   If they are in different tiles, the number of boundaries crossed is the cost.
    #   In row j, boundaries are at x = k where k+j is odd.
    #   The number of boundaries between sx and tx is the number of k such that
    #   min(sx, tx) < k <= max(sx, tx) and (k+j) is odd.
    
    # Let's use a more robust approach:
    # The distance is max(dy, (dx + 1) // 2) is for the case where you can move 
    # diagonally. Here you can't.
    # But you can move (0,0) -> (1,0) [free] -> (1,1) [cost 1] -> (2,1) [free].
    # This effectively moves you 2 units horizontally and 1 unit vertically for cost 1.
    # This is like a knight's move but only for the "free" parts.
    # The distance is actually:
    # cost = max(dy, (dx + 1) // 2)
    # Let's check Sample 2: sx=3, sy=1, tx=4, ty=1. dx=1, dy=0.
    # (3+1)%2 == 0, so A_{3,1} and A_{4,1} are same tile. Cost 0.
    # My formula max(0, (1+1)//2) = 1. Incorrect.
    
    # Correct logic:
    # Let's normalize coordinates so sx <= tx.
    # If sy == ty:
    #   Count k in (sx, tx] such that k+sy is odd.
    #   This is the number of boundaries.
    # If sy != ty:
    #   The cost is max(dy, (dx + 1) // 2) is almost correct.
    #   Let's use the property:
    #   To move dx horizontally and dy vertically, you can cover 2*dy horizontal 
    #   distance using the "staggered" free moves.
    #   The cost is dy + max(0, (dx - (dy + 1) + 1) // 2) ? 
    #   Let's re-evaluate.
    #   Each vertical step costs 1 and allows you to move 1 unit horizontally for free.
    #   Total horizontal distance covered by dy vertical steps is dy + 1 (including the start tile).
    #   Wait, if you move dy steps vertically, you can cover dx = dy + 1 horizontal units 
    #   by alternating (free horizontal, vertical, free horizontal...).
    #   Example: (0,0) -> (1,0) [free], (1,0) -> (1,1) [1], (1,1) -> (2,1) [free], 
    #   (2,1) -> (2,2) [1], (2,2) -> (3,2) [free].
    #   Here dy=2, dx=3, cost=2.
    #   If dx > dy + 1, the remaining distance is dx - (dy + 1).
    #   Each additional 2 units of dx costs 1 more (by moving L/R in the same row).
    #   So cost = dy + (dx - (dy + 1) + 1) // 2.
    #   This simplifies to: if dx <= dy + 1, cost = dy. Else, cost = dy + (dx - dy - 1 + 1) //