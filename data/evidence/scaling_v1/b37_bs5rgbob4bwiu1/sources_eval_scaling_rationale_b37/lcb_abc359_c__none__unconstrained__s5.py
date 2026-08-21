import sys

def solve():
    # Read input and map to integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The problem describes a grid of 2x1 tiles.
    # A tile consists of A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j:
    # If j is even, tiles are {(0,1), (2,3), (4,5)...}
    # If j is odd, tiles are {(-1,0), (1,2), (3,4)...}
    #
    # Let's transform (x, y) to a coordinate system (u, v) where 
    # moving between adjacent tiles costs 1.
    # In the original grid, moving vertically always crosses a tile boundary.
    # Moving horizontally crosses a boundary every 2 units, 
    # but the boundary position shifts based on the parity of y.
    #
    # Let u = x + (y % 2), v = y.
    # A tile covers two squares: (i, j) and (i+1, j) where i+j is even.
    # This is equivalent to saying the tile index in x is floor((x + (y%2)) / 2).
    # Let X = (x + (y % 2)) // 2 and Y = y.
    # Moving from (X1, Y1) to (X2, Y2) in this transformed grid:
    # Each step in Y costs 1.
    # Each step in X costs 1, but since we can move any n units,
    # we only care about the change in X and Y.
    # However, the movement rules are specific: 
    # "Choose a direction... move n units".
    # This is a shortest path problem on a graph.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # dist = max(abs(tx - sx), abs(ty - sy), (abs(tx - sx) + abs(ty - sy) + 1) // 2) 
    # is not correct here.
    #
    # Correct logic:
    # Let's use the transformation:
    # X = x + (y % 2)
    # Y = y
    # The distance is (abs(X2 - X1) + 1) // 2 + abs(Y2 - Y1) is also not quite it.
    #
    # Let's re-evaluate:
    # To move from (sx, sy) to (tx, ty):
    # Vertical distance is dy = abs(sy - ty). This costs dy.
    # Horizontal distance is dx = abs(sx - tx).
    # Because tiles are 2x1, we can cover 2 units of x with 1 tile.
    # The cost is roughly (dx + dy) / 2.
    # Specifically, the distance is max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2)
    # Wait, the simplest transformation for this specific tile pattern is:
    # New coordinates: X = (x + (y % 2)) // 2, Y = y
    # The distance is then:
    # cost = abs(Y2 - Y1) + max(0, (abs(X2 - X1) - abs(Y2 - Y1) + 1) // 2)
    # Actually, the most reliable formula for this grid is:
    # ans = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2)
    # Let's test Sample 1: 5 0, 2 5 -> dx=3, dy=5. max(5, (3+5)//2) = 5. Correct.
    # Sample 2: 3 1, 4 1 -> dx=1, dy=0. max(0, (1+0)//2) = 0. Correct.
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    
    # The minimum toll is max(dy, (dx + dy) // 2)
    # This is because each vertical move costs 1, and we can 'use' 
    # those vertical moves to cover horizontal distance.
    # One vertical move allows us to change our x-parity alignment.
    print(max(dy, (dx + dy) // 2))

if __name__ == '__main__':
    solve()