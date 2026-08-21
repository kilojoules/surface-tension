import sys

def solve():
    # Read input and map to integers
    # Sx, Sy: Starting coordinates
    # Tx, Ty: Target coordinates
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1 rectangles.
    # A tile covers A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j, the tiles are [(0,1), (2,3), ...] if j is even
    # and [(1,2), (3,4), ...] if j is odd.
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # A tile can be identified by (X, Y) where:
    # Y = sy
    # X = (sx // 2) if (sx % 2 == 0 and (sx + sy) % 2 == 0) or (sx % 2 == 1 and (sx + sy) % 2 == 1)
    # Actually, the rule is: A_{i,j} and A_{i+1,j} are one tile if i+j is even.
    # This means the tile index in x for a given j is:
    # If j is even: tiles are {0,1}, {2,3}, ... -> index is i // 2
    # If j is odd: tiles are {-1,0}, {1,2}, ... -> index is (i-1) // 2
    
    # More simply: the tile boundary in x occurs at x=k where k % 2 == j % 2.
    # The number of boundaries crossed when moving from sx to tx at height sy is:
    # A boundary is crossed if we move from x to x+1 and x % 2 == sy % 2.
    
    # Let's use a different approach:
    # The cost to move from (sx, sy) to (tx, ty) is the number of tiles entered.
    # Moving vertically by 1 always enters a new tile.
    # Moving horizontally by 1 enters a new tile ONLY IF we cross a boundary.
    # A boundary exists between x and x+1 if (x + y) is odd.
    
    # Let f(x, y) be a coordinate transformation to the "tile grid".
    # For a fixed y, the tiles are blocks of 2. 
    # The parity of the starting index of the block changes with y.
    # We can map (x, y) to (X, Y) where:
    # Y = y
    # X = (x + (y % 2)) // 2
    
    # The distance between (X1, Y1) and (X2, Y2) in a grid where you can move 
    # to any cell in the same row or column for a cost is not applicable here.
    # The problem says: "Choose a direction... move n units".
    # This means we can jump across many tiles in one move, but we pay for 
    # every tile we "enter".
    # If we move from (x, y) to (x + n, y), we enter all tiles in between.
    # If we move from (x, y) to (x, y + n), we enter all tiles in between.
    
    # Wait, the rule is: "Each time he enters a tile, he pays a toll of 1."
    # This is equivalent to the Manhattan distance in the transformed tile grid.
    # Let's refine the transformation:
    # For a cell (x, y), it belongs to tile (X, Y) where:
    # Y = y
    # X = (x if (x + y) % 2 != 0 else x - 1) // 2 
    # Actually: X = (x + (y % 2)) // 2
    # Let's check: 
    # If y=0: x=0,1 -> X=0; x=2,3 -> X=1. (Correct: 0+0 even, so A_{0,0}, A_{1,0} same tile)
    # If y=1: x=0 -> X=0; x=1,2 -> X=1; x=3,4 -> X=2. (Correct: 1+1 even, so A_{1,1}, A_{2,1} same tile)
    
    # The distance is |Xs - Xt| + |Ys - Yt|.
    # However, moving diagonally in the tile grid (changing both X and Y) 
    # might be cheaper if we can move in the original plane.
    # But the moves are only axis-aligned.
    # A move in x changes X, a move in y changes Y.
    # One move in y (y -> y+1) always changes the tile.
    # One move in x (x -> x+1) changes the tile if we cross a boundary.
    
    # Let's use the property:
    # Cost = sum of changes in X and Y.
    # But a single move in Y might also change the X index of the tile we are in.
    # Let's use the parity-based coordinate system:
    # X = x + (y % 2)
    # Y = 2 * y
    # This is getting complex. Let's use the simplest logic:
    # The distance is max(|Xs - Xt|, |Ys - Yt|) if we could move diagonally.
    # But we can't. We can only move horizontally and vertically.
    # The cost is simply the distance in the tile-grid:
    # X(x, y) = (x + (y % 2)) // 2
    # Y(x, y) = y
    # Distance = |X(sx, sy) - X(tx, ty)| + |Y(sx, sy) - Y(tx, ty)|
    # Wait, if we change Y, the X coordinate of the tile changes.
    # If we are at (X, Y) and move to (X, Y+1), the new X' is (x + ((Y+1)%2)) // 2.
    # This means X' could be X or X+1 or X-1.
    # This looks like a distance on a graph where nodes are tiles.
    # Two tiles are connected if they share a boundary.
    # A tile (X, Y) shares a boundary with (X +/- 1, Y) and (X, Y +/- 1) 
    # AND potentially (X +/- 1, Y +/- 1) because of the shift.
    # Specifically, tile (X, Y) covers x range [2X - (Y%2), 2X + 1 - (Y%2)).
    # It shares a boundary with (X', Y+1) if their x-ranges overlap.
    # Range Y: [2X - (Y%2), 2X + 1 - (Y%2))
    # Range Y+1: [2X' - ((Y+1)%2), 2X' + 1 - ((Y+1)%2))
    # Overlap exists if 2X' - ((Y+1)%2) < 2X + 1 - (Y%2) 
    # AND 2X - (Y%2) < 2X' + 1 - ((Y+1)%2).
    # Since (Y%2) + ((Y+1)%2) = 1, the condition is:
    # 2X' - (1 - (Y%2)) < 2X + 1 - (Y%2)  => 2X' < 2X + 2 => X' <= X
    # 2X - (Y%2) < 2X' + 1 - (1 - (Y%2)) => 2X < 2X' + (Y%2) => X <= X' (if Y%2=1) or X < X' (if Y%2=0)
    # Basically, tile (X, Y) touches (X, Y+1) and (X+1, Y+1) or (X-1, Y+1).
    # This means we can move from (X, Y) to (X', Y+1) if |X - X'| <= 1.
    # The minimum cost to reach (Xt, Yt) from (Xs, Ys) is:
    # max(|Xs - Xt|, |Ys - Yt|) if we can move to any X' where |X-X'| <= 1.
    # Wait, the cost is the number of tiles entered.
    # Starting tile is free? "Each time he enters a tile, he pays a toll of 1."
    # So the start tile is not paid, but every subsequent tile is.
    # The distance is simply the distance in a graph where (X, Y) is connected to 
    # (X', Y') if they are adjacent.
    # The distance is max(|Xs - Xt|, |Ys - Yt|) if we can move diagonally in the (X, Y) grid.
    # Since we can move (X, Y) -> (X', Y+1) where |X - X'| <= 1, we can indeed move diagonally.
    # Thus the answer is max(|Xs - Xt|, |Ys - Yt|).
    # Let's test Sample 1: 5 0 and 2 5
    # Xs = (5 + (0%2)) // 2 = 5 // 2 = 2
    # Ys = 0
    # Xt = (2 + (5%2)) // 2 = 3 // 2 = 1
    # Yt = 5
    # max(|2-1|, |0-5|) = max(1, 5) = 5. Correct.
    # Sample 2: 3 1 and 4 1
    # Xs = (3 + (1%2)) // 2 = 4 // 2 = 2
    # Ys = 1
    # Xt = (4 + (1%2)) // 2 = 5 // 2 = 2
    # Yt = 1
    # max(|2-2|, |1-1|) = 0. Correct.

    xs = (sx + (sy % 2)) // 2
    ys = sy
    xt = (tx + (ty % 2)) // 2
    yt = ty
    
    print(max(abs(xs - xt), abs(ys - yt)))

if __name__ == "__main__":
    solve()