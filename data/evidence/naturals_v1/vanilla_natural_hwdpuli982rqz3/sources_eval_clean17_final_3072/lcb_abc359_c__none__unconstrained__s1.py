```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are horizontal pairs starting at even i (if j is even)
    # or odd i (if j is odd).
    # Specifically, A_{i,j} and A_{i+1,j} are the same tile iff (i + j) % 2 == 0.
    
    # Let's transform the coordinates to a system where the "cost" is easier to calculate.
    # Notice that moving vertically always crosses a boundary between tiles (since tiles are 2x1).
    # Moving horizontally might stay within the same tile or cross into a new one.
    
    # Let x' = x + y and y' = x - y.
    # However, a simpler observation:
    # To get from (sx, sy) to (tx, ty), the minimum cost is related to the 
    # Manhattan distance, but we can "skip" costs by moving horizontally 
    # within a 2x1 tile.
    
    # Let's define the cost to reach (tx, ty) from (sx, sy).
    # The vertical distance is |sx - tx| + |sy - ty|.
    # But we can optimize.
    
    # Let dx = tx - sx and dy = ty - sy.
    # The number of horizontal boundaries crossed is |dy|.
    # The number of vertical boundaries crossed is |dx|.
    # A vertical boundary is at x = k. It is crossed if we move from x to x+1.
    # The boundary between A_{i,j} and A_{i+1,j} is NOT a tile boundary if i+j is even.
    
    # Minimum cost is actually:
    # cost = (|sx - tx| + |sy - ty|) // 2
    # But we must account for the parity of the starting and ending tiles.
    
    # Let's use the property:
    # A point (x, y) belongs to a tile. 
    # If (x + y) is even, (x, y) and (x+1, y) are in the same tile.
    # If (x + y) is odd, (x, y) and (x-1, y) are in the same tile.
    # Essentially, the tile ID is ( (x + (x+y)%2), y ).
    
    # The distance between two tiles (X1, Y1) and (X2, Y2) in this grid
    # where X is the index of the 2x1 block and Y is the row index:
    # The cost is max(|X1 - X2|, |Y1 - Y2|) if we could move diagonally.
    # But we move orthogonally.
    # The actual formula for this specific tiling problem is:
    # cost = (|sx - tx| + |sy - ty| + 1) // 2
    # However, we must check if the start and end points are in the same tile.
    
    # Let's refine:
    # Let f(x, y) = (x + y) // 2, g(x, y) = (x - y) // 2
    # The distance is max(|f(sx, sy) - f(tx, ty)|, |g(sx, sy) - g(tx, ty)|)
    # This is a known property for this specific "brick wall" tiling.
    
    f_s, g_s = (sx + sy) // 2, (sx - sy) // 2
    f_t, g_t = (tx + ty) // 2, (tx - ty) // 2
    
    # The cost is the Manhattan distance in the transformed coordinate system
    # but since we can move in 4 directions, the "toll" is accumulated 
    # whenever we enter a new tile.
    # The minimum toll is actually:
    # dist = max(|(sx+sy)//2 - (tx+ty)//2|, |(sx-sy)//2 - (tx-ty)//2|)
    # But we must be careful with integer division.
    
    # Let's use the property: 
    # Cost = (|sx - tx| + |sy - ty|) / 2, rounded up, 
    # unless the two points are in the same tile.
    
    # Correct logic for this specific problem:
    # The distance is max(|(sx + sy)//2 - (tx + ty)//2|, |(sx - sy)//2 - (tx - ty)//2|)
    # Wait, that's for 8-connectivity. For 4-connectivity:
    # The cost is (|sx - tx| + |sy - ty|) // 2.
    # Let's test Sample 1: 5 0, 2 5 -> |5-2| + |0-5| = 3 + 5 = 8. 8 // 2 = 4.
    # Sample 1 output is 5. So it's not just // 2.
    
    # Let's re-evaluate.
    # Each step in Y costs 1.
    # Each step in X costs 1, UNLESS we are moving between A_{i,j} and A_{i+1,j} where i+j is even.
    # This is equivalent to:
    # Cost = sum_{i=min(sx, tx)}^{max(sx, tx)-1} [ (i + y_current) % 2 != 0 ] + |sy - ty|
    # To minimize this, we can pick when to change y.
    
    # The minimum cost is actually:
    # ceil( (|sx - tx| + |sy - ty|) / 2 )
    # Let's check Sample 1: (3 + 5 + 1) // 2 = 4. Still not 5.
    
    # Let's use the coordinate transformation:
    # u = x + y, v = x - y
    # A move in x changes u by 1, v by 1.
    # A move in y changes u by 1, v by -1.
    # The tiles are defined by (u // 2, v).
    # Moving from (u, v) to (u+1, v+1) or (u-1, v-1) changes u//2 or v.
    # The cost is actually:
    # cost = max(|(sx + sy + 1)//2 - (tx + ty + 1)//2|, |(sx - sy + 1)//2 - (tx - ty + 1)//2|)
    # No, that's not it either.
    
    # Let's use the property:
    # The distance is (|sx - tx| + |sy - ty|) // 2, but if the parity of 
    # (sx + sy) and (tx + ty) are different, we might need an extra 1.
    # Actually, the most reliable way to think about this is:
    # Each tile covers two squares. One square is (i, j) where i+j is even, 
    # and the other is (i+1, j).
    # Let's map each square (i, j) to a tile index (I, J).
    # If (i + j) is even, the tile is (i // 2, j).
    # If (i + j) is odd, the tile is ((i - 1) // 2, j).
    # This is equivalent to: I = (i + (i + j) % 2) // 2, J = j.
    
    # Let's re-map:
    # For a square (i, j), it belongs to tile ( (i + (i+j)%2)//2, j )
    # Let X = (i + (i+j)%2)//2, Y = j.
    # Moving from (i, j) to (i+1, j):
    # If i+j is even, X stays same, Y stays same. Cost 0.
    # If i+j is odd, X increases by 1, Y stays same. Cost 1.
    # Moving from (i, j) to (i, j+1):
    # X might change, Y increases by 1. Cost 1.
    
    # This is a shortest path problem on a graph.
    # The cost to move from (X, Y) to (X+1, Y) is 1.
    # The cost to move from (X, Y) to (X, Y+1) is 1.
    # But wait, from (X, Y), we can reach (X, Y+1) and (X, Y-1) with cost 1.
    # Also, from (X, Y), we can reach (X+1, Y) and (X-1, Y) with cost 1.
    # Is there a way to move "diagonally" in (X, Y) space?
    # From (i, j), we can go to (i, j+1).
    # (i, j) -> (X, Y)
    # (i, j+1) -> ( (i + (i+j+1)%2)//2, j+1 )
    # If i+j is even, (i, j) is ( (i+0)//2, j ), (i, j+1) is ( (i+1)//2, j+1 ).
    # If i+j is odd, (i, j) is ( (i+1)//2, j ), (i, j+1) is ( (i+0)//2, j+1 ).
    
    # In both cases, |X_new - X| is either 0 or 1, and |Y_new - Y| is 1.
    # The cost to move from (X, Y) to (X', Y') is:
    # Since we can move from (X, Y) to (X, Y+1) or (X+/-1, Y+1) with cost 1,
    # the distance is max(|X_s - X_t|, |Y_s - Y_t|).
    
    # Let's test Sample 1: S(5, 0), T(2, 5)
    # S: i=5, j=0. i+j=5 (odd). X = (5+1)//2 = 3, Y = 0.
    # T: i=2, j=5. i+j=7 (odd). X = (2+1)//2 = 1, Y = 5.
    # cost = max(|3-1|, |0-5|) = max(2, 5) = 5. Correct!
    
    # Sample 2: S(3, 1), T(4, 1)
    # S: i=3, j=1. i+j=4 (even). X = (3+0)//2 = 1, Y = 1.
    # T: i=4, j=1. i+j=5 (odd). X = (4+1)//2 = 2, Y = 1.
    # Wait, (3, 1) and (4, 1) are in the same tile?
    # i=3, j=1: i+j=4 (even). Tile is A_{3,1} and A_{4,1}.
    # Yes, they are the same tile.
    # Let's use the formula:
    # S: i=3, j=1. i+j=4. X = (3+0)//2 = 1, Y = 1.
    # T: i=4, j=1. i+j=5. X = (4+1)//2 = 2, Y = 1.
    # max(|1-2|, |1-1|) = 1. But sample output is 0.
    
    # The formula max(|Xs-Xt|, |Ys-Yt|) is for the number of tiles entered.
    # The starting tile is already "entered".
    # The number of NEW tiles entered is the distance.
    # But we need to be careful: if S and T are in the same tile, cost is 0.
    # If they are in different tiles, the distance in the (X, Y) grid is the number of edges.
    # The number of edges is the number of new tiles.
    
    # Let's re-calculate X, Y:
    # For square (i, j):
    # If (i + j) % 2 == 0: tile is (i // 2, j) if we assume the pair is (i, j) and (i+1, j)
    # Let's use: Tile ID = ( (i + (i+j)%2) // 2, j