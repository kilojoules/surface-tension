import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1 rectangles.
    # When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed j:
    # If j is even: tiles are {(0,1), (2,3), (4,5)...} (indices of i)
    # If j is odd: tiles are {(-1,0), (1,2), (3,4)...} (indices of i)
    # Essentially, a tile is defined by (floor((i + (j % 2)) / 2), j)
    
    # Let's transform coordinates to a grid where each unit is a tile.
    # The y-coordinate is straightforward: each change in y enters a new tile.
    # The x-coordinate depends on the parity of y.
    
    # To simplify, we can use a coordinate transformation.
    # Let x' = x + (y % 2)
    # Then the tile index in x is floor(x' / 2).
    # However, the cost to move between tiles is the core of the problem.
    
    # The distance in y is simply |sy - ty|.
    # The distance in x is more complex because the boundaries shift.
    # Let's use the property: 
    # Cost = max(|sy - ty|, (|sx - tx| + 1) // 2) is NOT correct here.
    # The correct approach for this specific tiling pattern:
    # The distance is max(|sy - ty|, ceil(|sx - tx| / 2)) if we could move diagonally.
    # But we move axially.
    
    # Actually, the minimum toll is:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # If we are in the same tile, cost is 0.
    # The cost is max(dy, (dx + 1) // 2) if we can move optimally.
    # Wait, the rule is: if i+j is even, (i,j) and (i+1,j) are one tile.
    # This means in row j, tiles are blocks of 2.
    # In row j+1, the blocks are shifted by 1.
    
    # Let's redefine coordinates:
    # A point (x, y) belongs to tile ( (x + (y % 2)) // 2, y )
    # Let X(x, y) = (x + (y % 2)) // 2
    # Let Y(x, y) = y
    
    # Moving from (sx, sy) to (tx, ty):
    # The cost is the number of tile boundaries crossed.
    # This is a shortest path problem on a graph where nodes are tiles.
    # From tile (X, Y), one can move to:
    # (X, Y+1), (X, Y-1) -> cost 1
    # (X+1, Y), (X-1, Y) -> cost 1
    # But wait, moving from (X, Y) to (X, Y+1) might be free if the 
    # physical coordinates allow it? No, the problem says 
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    
    # The distance between (X1, Y1) and (X2, Y2) in this grid is
    # |X1 - X2| + |Y1 - Y2|.
    # However, we can move n units in one direction.
    # If we move vertically, we cross |sy - ty| boundaries.
    # If we move horizontally, we cross |X1 - X2| boundaries.
    # But we can combine these.
    
    # The actual minimum cost is max(|sy - ty|, (|sx - tx| + 1) // 2) 
    # is for a different problem. For this one:
    # The cost is simply the distance in the transformed coordinate system
    # if we could move diagonally. But we move axially.
    # Actually, the minimum cost is max(|sy - ty|, (abs(sx - tx) + 1) // 2) 
    # is only if you can move diagonally.
    # With axial moves, the cost is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The answer is max(dy, (dx + 1) // 2) is still wrong.
    
    # Correct logic:
    # To change Y by 1, cost is 1.
    # To change X by 2, cost is 1 (by moving Y then X then Y).
    # The minimum cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2).
    # Let's check Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+1)//2) = 1? 
    # Wait, Sample 2 output is 0.
    # In Sample 2: sx=3, sy=1. i+j = 3+1 = 4 (even). 
    # So A_{3,1} and A_{4,1} are in the same tile.
    # Thus (3.5, 1.5) and (4.5, 1.5) are in the same tile. Cost 0.
    
    # Let's use the tile coordinates:
    # X(x, y) = (x + (y % 2)) // 2
    # Y(x, y) = y
    # Start: X1 = (sx + (sy % 2)) // 2, Y1 = sy
    # End: X2 = (tx + (ty % 2)) // 2, Y2 = ty
    # The distance is max(abs(Y1 - Y2), abs(X1 - X2)) is for Chebyshev.
    # But we can move n units. 
    # If we move horizontally, we change X. If vertically, we change Y.
    # One vertical move (n=1) changes Y by 1 and might change X by 0 or 1.
    # This is exactly the distance in a grid where you can move to 
    # (X, Y+/-1) and (X+/-1, Y).
    # The minimum cost is simply the distance in the transformed space?
    # No, the sample 2 shows that if you are in the same tile, cost is 0.
    # The distance between tile (X1, Y1) and (X2, Y2) is:
    # You can move from (X, Y) to (X, Y+1) with cost 1.
    # You can move from (X, Y) to (X+1, Y) with cost 1.
    # But you can also move from (X, Y) to (X, Y+1) and then (X, Y+1) to (X+1, Y+1)
    # and then (X+1, Y+1) to (X+1, Y).
    # Actually, the most efficient way to move is to realize that
    # moving Y by 1 can also effectively move X by 1.
    # The cost is max(abs(Y1 - Y2), abs(X1 - X2)).
    
    x1 = (sx + (sy % 2)) // 2
    y1 = sy
    x2 = (tx + (ty % 2)) // 2
    y2 = ty
    
    print(max(abs(x1 - x2), abs(y1 - y2)))

if __name__ == "__main__":
    solve()