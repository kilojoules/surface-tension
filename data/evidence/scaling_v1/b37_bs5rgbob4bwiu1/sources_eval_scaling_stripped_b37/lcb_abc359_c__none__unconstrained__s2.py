import sys

def solve():
    # Read input and parse integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The problem describes a grid of 2x1 tiles.
    # A tile covers A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j:
    # If j is even, tiles are {(0,1), (2,3), (4,5)...} (indices of i)
    # If j is odd, tiles are {(-1,0), (1,2), (3,4)...} (indices of i)
    # Essentially, a tile is defined by the pair (j, floor((i + (j % 2)) / 2))
    
    # Let's transform the coordinates to a coordinate system where 
    # each unit represents a tile boundary.
    # For a fixed y, the tiles are blocks of 2 units wide.
    # The parity of the shift depends on y % 2.
    
    # We can use a coordinate transformation to map this to a 
    # standard grid distance problem.
    # Let u = x + (y % 2)
    # Let v = y
    # In the (u, v) space, a tile is a 2x1 block.
    # Specifically, a tile is {(u, v), (u+1, v)} where u is even.
    # Moving vertically (changing v) always enters a new tile.
    # Moving horizontally (changing u) only enters a new tile if we cross an even boundary.
    
    # The distance is effectively the number of tile boundaries crossed.
    # Vertical distance: |ty - sy|
    # Horizontal distance: 
    # We are at (sx, sy) and want to reach (tx, ty).
    # Let's use the property that the cost is related to the 
    # Manhattan distance in a transformed space.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # max(|sx - tx|, |sy - ty|) is NOT correct here because we can move 
    # n units. The actual cost is:
    # cost = abs(sy - ty) + (abs(sx - tx) + (something related to parity)) // 2
    
    # Correct logic for this specific tiling:
    # The distance is max(|sy - ty|, (|sx - tx| + |sy - ty| + 1) // 2) 
    # Wait, that's for a different tiling. 
    # Let's use the property: 
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # This simplifies to:
    # Cost = (abs(sx - tx) + abs(sy - ty) + 1) // 2 if we can move diagonally.
    # But we move in 4 directions.
    
    # Let's re-evaluate:
    # To move from (sx, sy) to (tx, ty):
    # Vertical steps always cost 1.
    # Horizontal steps cost 1 every 2 units, but the offset shifts.
    # This is equivalent to the distance in a grid where you can move 
    # to 8 neighbors, but vertical/horizontal costs differ.
    # Actually, the minimum cost is simply:
    # abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Wait, the simplest form is:
    # ans = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: 5 0 to 2 5. 
    # sx=5, sy=0, tx=2, ty=5.
    # abs(5-2)=3, abs(0-5)=5.
    # max(5, (3+5+1)//2) = max(5, 4) = 5. Correct.
    # Sample 2: 3 1 to 4 1.
    # abs(3-4)=1, abs(1-1)=0.
    # max(0, (1+0+1)//2) = 1. Wait, Sample 2 says 0.
    # Why? Because (3,1) and (4,1) are in the same tile.
    # i=3, j=1. i+j = 4 (even). So A_{3,1} and A_{4,1} are one tile.
    # My formula gives 1, but the answer is 0.
    
    # Let's use the parity logic:
    # A tile is identified by (j, (i + (j % 2)) // 2)
    # Let X(i, j) = (i + (j % 2)) // 2
    # Let Y(i, j) = j
    # The distance between (X1, Y1) and (X2, Y2) in this transformed grid
    # where you can move to any of the 8 neighbors (Chebyshev distance)
    # is max(abs(X1 - X2), abs(Y1 - Y2)).
    # However, the cost of a move is 1.
    # Moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1.
    # Moving from (X, Y) to (X+1, Y+1) costs 1.
    # This is exactly the Chebyshev distance!
    
    # Let's test Sample 2: (3, 1) and (4, 1)
    # X1 = (3 + (1 % 2)) // 2 = (3 + 1) // 2 = 2
    # X2 = (4 + (1 % 2)) // 2 = (4 + 1) // 2 = 2
    # Y1 = 1, Y2 = 1
    # max(abs(2-2), abs(1-1)) = 0. Correct.
    
    # Sample 1: (5, 0) and (2, 5)
    # X1 = (5 + (0 % 2)) // 2 = 5 // 2 = 2
    # X2 = (2 + (5 % 2)) // 2 = (2 + 1) // 2 = 1
    # Y1 = 0, Y2 = 5
    # max(abs(2-1), abs(0-5)) = max(1, 5) = 5. Correct.
    
    # Final Formula:
    # X = (i + (j % 2)) // 2
    # Y = j
    # Distance = max(abs(X1 - X2), abs(Y1 - Y2))
    
    x1 = (sx + (sy % 2)) // 2
    y1 = sy
    x2 = (tx + (ty % 2)) // 2
    y2 = ty
    
    print(max(abs(x1 - x2), abs(y1 - y2)))

if __name__ == "__main__":
    solve()