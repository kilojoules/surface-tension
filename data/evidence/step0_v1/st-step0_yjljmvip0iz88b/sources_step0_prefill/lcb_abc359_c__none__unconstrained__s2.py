```python
import sys

def solve():
    # Read input from standard input
    try:
        line = sys.stdin.read().split()
        if not line:
            return
        sx, sy, tx, ty = map(int, line)
    except EOFError:
        return

    # The rule for tiling is:
    # If i + j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed row j:
    # If j is even, pairs (0,j)-(1,j), (2,j)-(3,j), ... are tiles.
    # If j is odd, pairs (1,j)-(2,j), (3,j)-(4,j), ... are tiles.
    # In general, in row j, the tile boundaries are at x = k where k % 2 == (j % 2).
    
    # Let's define a coordinate transformation to make the grid uniform.
    # In the original grid, moving vertically always crosses a tile boundary 
    # unless you stay within the same 2x1 tile (which is impossible for vertical moves).
    # Moving horizontally might or might not cross a boundary.
    
    # Let's analyze the cost.
    # To move from (sx, sy) to (tx, ty):
    # The vertical distance is |sy - ty|. Each vertical step of 1 unit 
    # necessarily enters a new tile.
    # The horizontal distance is |sx - tx|. 
    # In a row j, boundaries are at x such that x % 2 == j % 2.
    
    # Let's simplify the problem by transforming coordinates.
    # Let x' = x, y' = y.
    # The cost to move from (sx, sy) to (tx, ty) is the minimum number of 
    # tile boundaries crossed.
    # A move is a straight line. A sequence of moves is allowed.
    # This is equivalent to finding the shortest path in a graph where 
    # nodes are tiles and edges exist between adjacent tiles.
    
    # Let's define the tile ID for square (i, j):
    # If (i + j) is even, square (i, j) and (i+1, j) are the same tile.
    # We can represent the tile ID as ( (i if (i+j)%2 != 0 else i-1), j ) 
    # But it's easier to say: in row j, the tile index is floor((i + (j%2)) / 2).
    
    # Let X(i, j) = (i + (j % 2)) // 2
    # Let Y(i, j) = j
    # The distance between (sx, sy) and (tx, ty) in this transformed space:
    # The cost to move from (X1, Y1) to (X2, Y2) is:
    # Each change in Y costs 1.
    # Each change in X costs 1.
    # However, we can move diagonally in the original space by combining 
    # a horizontal and vertical move.
    # Actually, the problem is simpler:
    # The cost is max(|X1 - X2|, |Y1 - Y2|) if we could move diagonally.
    # But we can only move L, R, U, D.
    # Wait, the rule is: "Choose a direction... and a positive integer n."
    # This means we can move any distance in one of the 4 cardinal directions.
    # This is equivalent to Manhattan distance in the tile-graph.
    # But we can move n units. If we move n units vertically, we cross n tiles.
    # If we move n units horizontally, we might cross several tiles.
    
    # Let's re-evaluate.
    # In row j, tiles are blocks of 2. 
    # To get from (sx, sy) to (tx, ty):
    # Let's use the transformation:
    # u = x + y
    # v = x - y
    # This is a common trick for Manhattan distance, but here the grid is shifted.
    
    # Let's use the property:
    # The cost is the distance in the graph of tiles.
    # Two tiles are adjacent if they share an edge.
    # In row j, tile T_{k,j} is adjacent to T_{k-1,j} and T_{k+1,j}.
    # T_{k,j} is also adjacent to tiles in row j-1 and j+1 that it touches.
    # Tile T_{k,j} covers x-range [2k - (j%2), 2k + 1 - (j%2)] and y-range [j, j+1].
    # It touches tiles in row j+1 that cover x-range [2k - (j%2), 2k + 1 - (j%2)].
    # In row j+1, boundaries are at x % 2 == (j+1)%2.
    # The x-range of T_{k,j} is length 2. It will always cover exactly one 
    # boundary of row j+1, meaning it touches exactly 2 tiles in row j+1.
    
    # This structure is exactly a grid rotated by 45 degrees.
    # The distance between (X1, Y1) and (X2, Y2) in this specific tile-graph 
    # is known to be:
    # cost = max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) / 2
    # Wait, that's for a different grid. Let's derive it.
    
    # Let's use the transformation:
    # z1 = x + y
    # z2 = x - y
    # In our case, the tiles are 2x1.
    # The distance is actually:
    # dist = max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) 
    # But we must account for the 2x1 nature.
    # The correct distance for this specific tiling is:
    # cost = max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) // 2
    # Let's check Sample 1: (5,0) to (2,5)
    # sx+sy = 5, tx+ty = 7. Diff = 2.
    # sx-sy = 5, tx-ty = -3. Diff = 8.
    # max(2, 8) // 2 = 4. Sample output says 5. My formula is wrong.
    
    # Let's try: cost = (|sx - tx| + |sy - ty| + 1) // 2 ? 
    # Sample 1: (3 + 5 + 1) // 2 = 4. Still wrong.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # You must change y by |sy - ty|. This costs |sy - ty| tolls.
    # While doing this, you can move horizontally.
    # In each step of y, you are in a new row. 
    # In row j, you can move to any x within the same tile for free.
    # The tiles in row j are [0,2), [2,4)... if j even, and [-1,1), [1,3)... if j odd.
    # This is like a graph where you can move from (x, y) to (x +/- 1, y) 
    # with cost 1 if you cross a boundary, and (x, y) to (x, y +/- 1) with cost 1.
    
    # The distance is actually:
    # dist = max(|sy - ty|, (|sx - tx| + |sy - ty| + 1) // 2)
    # Wait, let's test Sample 1: max(5, (3 + 5 + 1) // 2) = max(5, 4) = 5. Correct.
    # Sample 2: (3,1) to (4,1). max(0, (1 + 0 + 1) // 2) = 1. Incorrect. Sample 2 is 0.
    
    # Let's refine:
    # If sy == ty:
    #   Cost is 0 if sx and tx are in the same tile, else 1 if they are adjacent, etc.
    #   In row sy, boundaries are at x % 2 == sy % 2.
    #   Tile index is (x + (sy % 2)) // 2.
    #   Cost is |(sx + (sy % 2)) // 2 - (tx + (sy % 2)) // 2|.
    #   But we can move to row sy+1, move horizontally, and come back.
    #   That would cost 1 (up) + 1 (down) = 2.
    #   So cost is min(|Xs - Xt|, 2) if sy == ty? No, that's not right.
    
    # Let's use the property that this is a shortest path on a graph.
    # The distance is:
    # cost = max(|sy - ty|, (|sx - tx| + |sy - ty| + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # No, let's use the coordinate transform:
    # Let u = x + y, v = x - y.
    # The distance is max(|u1 - u2|, |v1 - v2|) // 2.
    # Let's check Sample 1: u1=5, v1=5; u2=7, v2=-3. max(2, 8) // 2 = 4. Still 4.
    
    # Let's try another approach.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # abs(sy - ty) + max(0, (|sx - tx| - abs(sy - ty) + 1) // 2)
    # Wait, if we move vertically, we can "gain" horizontal distance.
    # In each vertical step, we move from row j to j+1.
    # The boundaries shift. This allows us to move 1 unit horizontally 
    # for "free" every 2 vertical steps? No.
    
    # Let's look at the grid:
    # Row 0: [0,1][2,3][4,5]...
    # Row 1: [-1,0][1,2][3,4]...
    # Row 2: [0,1][2,3][4,5]...
    # To move from (0,0) to (1,0), cost is 0 (same tile).
    # To move from (0,0) to (2,0), cost is 1 (cross boundary at x=1).
    # To move from (0,0) to (1,1), cost is 1 (move up).
    # To move from (0,0) to (2,1), cost is 1 (move up, then move left/right within tile).
    # To move from (0,0) to (3,1), cost is 2 (move up, then cross one boundary).
    
    # Let's use the transformation:
    # X = x, Y = y
    # The distance is:
    # If sy == ty:
    #    cost = abs((sx + (sy%2))//2 - (tx + (sy%2))//2)
    # Else:
    #    # We can move vertically and horizontally.
    #    # The cost is max(|sy - ty|, (|sx - tx| + |sy - ty| + 1) // 2)
    #    # But we must check the parity of the start and end tiles.
    #    # Let's use the formula: 
    #    # dist = max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    #    # For Sample 2: sx=3, sy=1, tx=4, ty=1.
    #    # sy == ty, so cost = abs((3+1)//2 - (4+1)//2) = abs(2 - 2) = 0. Correct.
    #    # For Sample 1: sx=5, sy=0, tx=2, ty=5.
    #    # sy != ty, so cost = max(5, (3 + 5 + 1) // 2) = 5. Correct.
    
    # Let's double check the sy != ty case.
    # If we move from (0,0) to (0,1), cost is 1. Formula: max(1, (0+1+1)//2) = 1.
    # If we move from (0,0) to (1,1), cost is 1. Formula: max(1, (