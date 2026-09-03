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

    # The tiling rule:
    # A_{i,j} and A_{i+1,j} are in the same tile if i+j is even.
    # This means for a fixed j:
    # If j is even, tiles are (0,1), (2,3), (4,5)... (i is even)
    # If j is odd, tiles are (1,2), (3,4), (5,6)... (i is odd)
    
    # Let's define a coordinate transformation to a grid where each unit is a tile.
    # In the original grid, moving vertically always crosses a tile boundary 
    # unless you stay within the same 2x1 tile (which is impossible for vertical moves).
    # Moving horizontally might not cross a boundary.
    
    # Let's analyze the cost.
    # To move from (sx, sy) to (tx, ty):
    # The vertical distance is |sy - ty|. Each vertical step of 1 unit 
    # necessarily enters a new tile.
    # The horizontal distance is |sx - tx|. 
    # However, the "cost" of horizontal movement depends on the parity of the row.
    
    # Let's simplify the problem by transforming coordinates.
    # A tile covers {(i, j), (i+1, j)} if i+j is even.
    # This is equivalent to saying a tile is defined by (floor((i + (j%2))/2), j).
    # Let X(i, j) = (i + (j % 2)) // 2
    # Let Y(i, j) = j
    # The cost to move between two tiles (X1, Y1) and (X2, Y2) in a grid 
    # where you can move any distance in one direction is the 
    # L1 distance if you have to pay for every tile entered.
    # Wait, the rule is: "Each time he enters a tile, he pays a toll of 1."
    # If he is already in tile T and moves to tile T', he pays 1.
    # If he moves and stays within the same tile, he pays 0.
    
    # Let's re-evaluate.
    # From (sx, sy) to (tx, ty):
    # Let's use the transformation:
    # x' = (sx + (sy % 2)) // 2
    # y' = sy
    # The distance is then related to |x'_s - x'_t| and |y'_s - y'_t|.
    # Specifically, the cost is max(|x'_s - x'_t|, |y'_s - y'_t|) ? 
    # No, that's for 8-connectivity.
    
    # Let's look at the structure:
    # It's a grid of 2x1 tiles. 
    # In row j, tiles are horizontal. 
    # To move from (sx, sy) to (tx, ty):
    # The minimum cost is actually (abs(sx - tx) + abs(sy - ty)) // 2 
    # if we can move diagonally? No, we can only move L or R or U or D.
    
    # Let's use the property:
    # The cost is equivalent to the distance in a graph where nodes are tiles.
    # Two tiles are connected if they share a boundary.
    # The distance between tile (X1, Y1) and (X2, Y2) is |X1 - X2| + |Y1 - Y2|.
    # But we can move any distance n in one direction.
    # If we move horizontally, we cross |X1 - X2| tiles.
    # If we move vertically, we cross |Y1 - Y2| tiles.
    # However, a single vertical move of n units might cross multiple tiles.
    # Actually, the problem is simpler:
    # The cost is the distance in the L1 metric on the transformed grid:
    # X = (i + (j % 2)) // 2, Y = j
    # But we can move any n units. This means we can change X or Y in one move.
    # The cost to change X by dX is dX, and Y by dY is dY.
    # But we can combine them? No, the moves are axis-aligned.
    # The minimum cost to get from (Xs, Ys) to (Xt, Yt) is |Xs - Xt| + |Ys - Yt|.
    # Wait, the sample 1: (5,0) to (2,5).
    # Xs = (5 + 0)//2 = 2, Ys = 0
    # Xt = (2 + (5%2))//2 = (2+1)//2 = 1, Yt = 5
    # Cost = |2-1| + |0-5| = 1 + 5 = 6. Sample output says 5.
    # Why? Because one move can change both X and Y? No.
    # "Choose a direction (up, down, left, or right) and a positive integer n."
    # This means we can only change one coordinate at a time.
    # But a vertical move of n units crosses n tiles.
    # A horizontal move of n units crosses some number of tiles.
    
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # If he is in tile T and moves to T', he pays 1.
    # If he moves n units and passes through tiles T, T1, T2, ..., Tk, he pays k.
    # This is exactly the L1 distance in the tile-grid.
    # The only catch is that a vertical move of n units might enter n tiles,
    # but some of those might be the same tile? No, tiles are 2x1 (horizontal).
    # So a vertical move of n units always enters n new tiles.
    # A horizontal move of n units enters some number of tiles.
    # In row j, tiles are {(0,1), (2,3)...} if j even.
    # Moving from x=5 to x=2 in row 0:
    # Tiles are [0,1], [2,3], [4,5]. 
    # x=5.5 is in tile [4,5]. x=2.5 is in tile [2,3].
    # He enters tile [3,4] then tile [2,3]. Cost = 2.
    # Wait, the sample 1 says cost 5.
    # (5,0) -> (2,5). 
    # X-dist: (5,0) is in tile X=2, (2,0) is in tile X=1. Dist = 1.
    # Y-dist: 0 to 5. Dist = 5.
    # Total = 6. Still not 5.
    
    # Let's re-examine the "diagonal" possibility.
    # He can move Right, then Up, then Left...
    # If he is at the boundary of two tiles, he can enter either.
    # The key is that he can move to a point (x, y) and then move in another direction.
    # If he is at a corner where 4 tiles meet, he can pick the best one.
    # Actually, the minimum cost is simply:
    # cost = max(|Xs - Xt|, |Ys - Yt|) if we could move diagonally.
    # But we can't. However, we can move in a staircase pattern.
    # The cost is actually (abs(Xs - Xt) + abs(Ys - Yt) + 1) // 2 ? 
    # No, let's look at the coordinates again.
    # The distance is simply the L1 distance in the transformed grid, 
    # but you can "save" cost by moving to the boundary.
    # The correct answer for this specific tiling problem is:
    # cost = (abs(Xs - Xt) + abs(Ys - Yt)) 
    # But wait, the sample 1: Xs=2, Ys=0, Xt=1, Yt=5. Sum = 6.
    # The only way to get 5 is if we can move "diagonally" in the tile grid.
    # A move in the original plane is axis-aligned.
    # A move from (x, y) to (x, y+1) enters 1 tile.
    # A move from (x, y) to (x+1, y) enters 0 or 1 tile.
    # If we move (x, y) -> (x+1, y) -> (x+1, y+1), we enter 1 + 1 = 2 tiles.
    # If we move (x, y) -> (x, y+1) -> (x+1, y+1), we enter 1 + 0/1 tiles.
    # The trick is: if we are at a boundary, we can move horizontally for free 
    # if the current tile extends horizontally.
    # The minimum cost is actually:
    # abs(Ys - Yt) + (abs(Xs - Xt) + 1) // 2 is not it.
    # Let's use the formula: cost = abs(Ys - Yt) + max(0, (abs(Xs - Xt) - (abs(Ys - Yt) + 1) // 2) * 2) ? No.
    
    # Correct logic for this problem:
    # The distance is abs(Ys - Yt) + max(0, abs(Xs - Xt) - (abs(Ys - Yt) + 1) // 2 * 2) 
    # No, the simplest form is:
    # The cost is abs(Ys - Yt) + max(0, 2 * abs(Xs - Xt) - abs(Ys - Yt)) 
    # Let's check Sample 1: Xs=2, Ys=0, Xt=1, Yt=5.
    # abs(0-5) + max(0, 2*|2-1| - 5) = 5 + max(0, 2-5) = 5. Correct!
    # Sample 2: (3,1) to (4,1).
    # Xs = (3 + 1)//2 = 2, Ys = 1.
    # Xt = (4 + 1)//2 = 2, Yt = 1.
    # abs(1-1) + max(0, 2*|2-2| - 0) = 0. Correct!
    
    # Let's double check the logic.
    # Each vertical step of 1 unit costs 1 and allows us to potentially 
    # change our X-tile by 1 (because the tile boundaries shift).
    # So for every 2 vertical steps, we can effectively move 1 X-tile for free.
    # Wait, if we move Y by 1, the parity of Y changes, so the tile boundaries shift.
    # This means we can move horizontally across a boundary for free every other step.
    # The cost is abs(Ys - Yt) + max(0, 2 * abs(Xs - Xt) - abs(Ys - Yt))
    # This simplifies to: max(abs(Ys - Yt), 2 * abs(Xs - Xt))
    # Wait, let's check: 
    # Sample 1: max(5, 2*1) = 5.
    # Sample 2: max(0, 2*0) = 0.
    # Let's test another: Xs=0, Ys=0, Xt=2, Yt=0.
    # Cost: max(0, 2*2) = 4.
    # Manual: (0,0) tile 0 -> (1,0) tile 0 -> (2,0) tile 1 -> (3,0) tile 1 -> (4,0) tile 2.
    # To get to X=2, we need to enter tile 1 and tile 2. Cost = 2.
    # My formula says 4. Something is wrong.
    
    # Let's re-evaluate.
    # In row j, tiles are [0,2), [2,4)... if j even.
    # To move from X=0 to X=2 in row 0, you enter tile [2,4). Cost = 1.
    # So the cost to move from Xs to Xt in the same row is simply abs(Xs - Xt).
    # Then the total cost is abs(Ys - Yt) + abs(Xs - Xt).
    # But we can reduce this if we use the shifting boundaries.
    # If we move Y by 1, the X-coordinates of the tiles shift.
    # A move (X, Y) -> (X, Y+1) costs 1.
    # Now we are in row Y+1. The tile boundaries are shifted.
