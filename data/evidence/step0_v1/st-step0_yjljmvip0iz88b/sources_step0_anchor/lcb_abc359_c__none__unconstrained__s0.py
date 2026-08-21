import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, if j is even, tiles are [0,1], [2,3], ...
    # In row j, if j is odd, tiles are [-1,0], [1,2], ...
    
    # Let's define a coordinate transformation to a grid where each unit is a tile.
    # For a cell (x, y), the tile index in the x-direction depends on the parity of y.
    # If y is even, tile_x = x // 2
    # If y is odd, tile_x = (x - 1) // 2 = (x + 1) // 2 - 1
    # However, a simpler way to think about it:
    # The cost to move between two points in this specific tiling is:
    # cost = max(|tx - sx|, |ty - sy|) is NOT correct here because tiles are 2x1.
    # The actual distance is based on the fact that moving vertically always 
    # crosses a tile boundary, but moving horizontally might not.
    
    # Let's normalize the coordinates.
    # In any row y, the tiles are pairs (2k, 2k+1) if y is even, and (2k-1, 2k) if y is odd.
    # This is equivalent to saying the tile boundary is at x = k if (k + y) is odd.
    
    # The minimum cost to get from (sx, sy) to (tx, ty) in this grid is:
    # cost = ceil( abs(sx - tx) / 2 ) + abs(sy - ty)
    # But we must account for the offset of the tiles in each row.
    # If we move from (sx, sy) to (tx, ty), the vertical distance is always |sy - ty|.
    # The horizontal distance is trickier. 
    # Let's use the transformation: 
    # A point (x, y) belongs to tile ( (x if y%2==0 else x-1)//2 , y )
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # If j is even: i=0, 2, 4... are starts of tiles.
    # If j is odd: i=-1, 1, 3... are starts of tiles.
    
    # Let's map (x, y) to a coordinate (X, Y) in a grid of tiles.
    # If y is even, X = x // 2.
    # If y is odd, X = (x + 1) // 2. 
    # Wait, if y is odd, i+j is even when i is odd. So A_{1,j} and A_{2,j} are one tile.
    # So for y odd, the tiles are ..., [-1, 0], [1, 2], [3, 4] ...
    # The tile index for x when y is odd is (x+1) // 2.
    
    # Let's refine:
    # For a fixed y, the tile index is:
    # if y % 2 == 0: tx_idx = x // 2
    # if y % 2 == 1: tx_idx = (x + 1) // 2
    
    # The distance between (sx, sy) and (tx, ty) is:
    # The vertical cost is |sy - ty|.
    # The horizontal cost is |tx_idx_s - tx_idx_t|.
    # However, we can move diagonally in terms of tiles.
    # One move (n units) in one direction.
    # If we move vertically, we pay 1 per tile.
    # If we move horizontally, we pay 1 per tile.
    # This is Manhattan distance in the tile-grid.
    # But we can "cheat" by moving to a row where the horizontal distance is shorter.
    
    # The cost is actually:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # The minimum cost is (dx + 1) // 2 + dy if we just move horizontally then vertically.
    # But we can optimize. The correct formula for this specific tiling is:
    # cost = max(dy, (dx + dy + 1) // 2) 
    # No, that's for a different problem.
    
    # Let's use the property:
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2 - abs(sy - ty))
    # Actually, the simplest form is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's test Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+5+1)//2) = max(5, 4) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+0+1)//2) = 1. 
    # Wait, Sample 2 output is 0. My formula is wrong.
    
    # Re-evaluating Sample 2: (3, 1) and (4, 1).
    # i=3, j=1. i+j = 4 (even). So A_{3,1} and A_{4,1} are the same tile.
    # Thus, moving from 3.5 to 4.5 in row 1 costs 0.
    
    # Correct logic:
    # In row y, x is in tile (x // 2) if y is even, and ((x+1) // 2) if y is odd.
    # Let X(x, y) = x // 2 if y % 2 == 0 else (x + 1) // 2
    # The distance is the Manhattan distance in the tile grid:
    # cost = abs(X(sx, sy) - X(tx, ty)) + abs(sy - ty)
    # But we can move to a different row first.
    # The cost is min(
    #    abs(X(sx, sy) - X(tx, ty)) + abs(sy - ty),
    #    abs(X(sx, sy) - X(tx, ty+1)) + abs(sy - (ty+1)),
    #    ...
    # )
    # Actually, the cost is simply:
    # cost = abs(sy - ty) + max(0, abs(X(sx, sy) - X(tx, ty)) - abs(sy - ty))
    # Wait, if we move vertically, we can change our X coordinate relative to the tiles.
    # The most efficient way is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Let's check Sample 2: sx=3, sy=1, tx=4, ty=1.
    # (3+1)%2 = 0, (4+1)%2 = 1. Parity differs.
    # cost = max(0, (1 + 0 + 1) // 2) = 1. Still 1.
    
    # Let's use the X(x, y) logic.
    # sx=3, sy=1 -> X = (3+1)//2 = 2.
    # tx=4, ty=1 -> X = (4+1)//2 = 2.
    # cost = abs(2 - 2) + abs(1 - 1) = 0. Correct!
    
    # Now, can we do better by changing rows?
    # If we move from (sx, sy) to (tx, ty), we must pay at least |sy - ty|.
    # While moving vertically, we can pick whether to enter the tile at x or x+1.
    # The horizontal distance in tiles is |X(sx, sy) - X(tx, ty)|.
    # Each vertical move can potentially reduce the horizontal tile distance by 1.
    # So cost = abs(sy - ty) + max(0, abs(X(sx, sy) - X(tx, ty)) - abs(sy - ty))
    # This simplifies to: cost = max(abs(sy - ty), abs(X(sx, sy) - X(tx, ty)))
    # Wait, that's not right. If we move vertically, we are already paying for the tile.
    # The correct logic:
    # To get from (sx, sy) to (tx, ty):
    # 1. Vertical distance is dy = abs(sy - ty).
    # 2. Horizontal distance in tiles is dx_tiles = abs(X(sx, sy) - X(tx, ty)).
    # 3. Each vertical step moves us to a new tile. We can use that step to also 
    #    move horizontally.
    # 4. The total cost is max(dy, dx_tiles).
    # Let's check Sample 1: 5 0, 2 5.
    # X(5, 0) = 5 // 2 = 2.
    # X(2, 5) = (2 + 1) // 2 = 1.
    # dy = 5, dx_tiles = |2 - 1| = 1.
    # cost = max(5, 1) = 5. Correct.
    # Sample 2: 3 1, 4 1.
    # X(3, 1) = (3 + 1) // 2 = 2.
    # X(4, 1) = (4 + 1) // 2 = 2.
    # dy = 0, dx_tiles = 0.
    # cost = max(0, 0) = 0. Correct.
    
    # Final formula:
    # X_s = sx // 2 if sy % 2 == 0 else (sx + 1) // 2
    # X_t = tx // 2 if ty % 2 == 0 else (tx + 1) // 2
    # result = max(abs(sy - ty), abs(X_s - X_t))
    
    # Using a list comprehension to avoid assignment and then printing the result.
    # But the prompt allows variables, just not loops/recursion.
    
    x_s = (sx // 2) if sy % 2 == 0 else (sx + 1) // 2
    x_t = (tx // 2) if ty % 2 == 0 else (tx + 1) // 2
    print(max(abs(sy - ty), abs(x_s - x_t)))

if __name__ == "__main__":
    solve()