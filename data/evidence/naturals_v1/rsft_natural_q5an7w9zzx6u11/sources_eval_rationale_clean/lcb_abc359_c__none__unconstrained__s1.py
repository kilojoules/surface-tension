import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])
    
    # The tiles are 2x1 blocks. 
    # If i+j is even, (i, j) and (i+1, j) are one tile.
    # This means for a fixed j, tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    # Effectively, the tile boundaries for x are at x = k*2 + (j % 2).
    
    # Let's transform the coordinates to a system where we can calculate the cost.
    # The cost to move between two points in this grid is related to the 
    # Manhattan distance, but moving horizontally within a tile is free.
    # The cost is essentially the number of tile boundaries crossed.
    
    # For a fixed y, the x-boundaries are at x = 2k + (y % 2).
    # The distance in y is simply |sy - ty| because every vertical move 
    # crosses a tile boundary.
    # The distance in x depends on whether we are in the same tile.
    
    # A more robust approach:
    # The cost is max(|sx - tx|, |sy - ty|) if we could move diagonally, 
    # but we move orthogonally.
    # The actual cost is (|sx - tx| + |sy - ty|) / 2, adjusted for parity.
    
    # Let's use the derived formula for this specific tiling problem:
    # The cost is (|sx - tx| + |sy - ty|) // 2, but we must account for 
    # the specific offsets of the tiles.
    
    # Correct logic:
    # Let dx = |sx - tx| and dy = |sy - ty|
    # The cost is (dx + dy) // 2, but if (sx + sy) % 2 != (tx + ty) % 2
    # and we are moving between specific parity cells, it might vary.
    # Actually, the simplest correct formula for this grid is:
    # cost = max(0, (abs(sx - tx) + abs(sy - ty) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Wait, the sample 1: (5,0) to (2,5). dx=3, dy=5. (3+5)//2 = 4. Sample output is 5.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0. (1+0)//2 = 0. Sample output 0.
    
    # Re-evaluating:
    # In row j, tiles are [2k + (j%2), 2k + 2 + (j%2)].
    # To move from (sx, sy) to (tx, ty):
    # The cost is the number of tiles entered.
    # This is equivalent to the distance in a graph where nodes are tiles.
    # The distance between tile (i, j) and (i', j') is (|i - i'| + |j - j'|) 
    # where i is the tile index in the row.
    # In row j, square (x, j) belongs to tile floor((x - (j%2)) / 2).
    
    # Let's define the tile coordinates:
    # TileX(x, y) = (x - (y % 2)) // 2
    # TileY(x, y) = y
    
    # The distance between (TX1, TY1) and (TX2, TY2) in this grid 
    # (where you can move to adjacent tiles) is:
    # dist = |TY1 - TY2| + |TX1 - TX2|
    # However, moving vertically might land you in a tile that 
    # also covers the X coordinate you need.
    
    # The correct distance formula for this specific tiling is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5.
    # abs(0-5) = 5. (3 + 5 + 1) // 2 = 4. Max(5, 4) = 5. Correct.
    # Sample 2: sx=3, sy=1, tx=4, ty=1.
    # abs(1-1) = 0. (1 + 0 + 1) // 2 = 1. 
    # Wait, Sample 2 output is 0. 
    # In Sample 2: (3,1) and (4,1). i+j = 3+1=4 (even). 
    # Rule: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # So A_{3,1} and A_{4,1} are the same tile. Cost 0.
    
    # Let's refine:
    # Tile index for (x, y) is:
    # If (x + y) is even, it's the left half of a tile.
    # If (x + y) is odd, it's the right half of a tile.
    # Let's use a coordinate transformation:
    # Each tile can be identified by (X, Y) where:
    # Y = y
    # X = (x if (x+y)%2 != 0 else x) ... no.
    # If (x+y) is even, square (x,y) and (x+1,y) are one tile.
    # This tile can be indexed as X = x // 2 if (x+y)%2 == 0 else (x-1) // 2.
    # Actually: X = (x - (y % 2)) // 2.
    # Let's check Sample 2: (3,1) and (4,1).
    # X1 = (3 - (1%2)) // 2 = (3-1)//2 = 1.
    # X2 = (4 - (1%2)) // 2 = (4-1)//2 = 1.
    # Y1 = 1, Y2 = 1.
    # Distance = |1-1| + |1-1| = 0. Correct.
    
    # Sample 1: (5,0) and (2,5).
    # X1 = (5 - (0%2)) // 2 = 5 // 2 = 2.
    # Y1 = 0.
    # X2 = (2 - (5%2)) // 2 = (2-1) // 2 = 0.
    # Y2 = 5.
    # Distance = |2-0| + |0-5| = 2 + 5 = 7? 
    # No, the sample output is 5. 
    # The distance is not Manhattan because one move can change both X and Y.
    # A move from (X, Y) to (X, Y+1) is 1 unit.
    # But the tile at (X, Y+1) might be different from the tile at (X, Y).
    # The actual distance is:
    # dist = max(abs(Y1 - Y2), (abs(X1 - X2) + abs(Y1 - Y2) + 1) // 2) 
    # Wait, that's for a different grid.
    
    # Let's use the property:
    # To get from (X1, Y1) to (X2, Y2):
    # Each vertical step changes Y by 1 and potentially X by 0 or 1.
    # The cost is max(|Y1 - Y2|, ceil(|X1 - X2| / 1)) if we can move diagonally.
    # But we move orthogonally.
    # The correct distance is:
    # cost = abs(Y1 - Y2) + max(0, abs(X1 - X2) - (abs(Y1 - Y2) + 1) // 2) 
    # No.
    
    # Let's use the logic: 
    # To move from (X1, Y1) to (X2, Y2), we must cover |Y1 - Y2| vertical distance.
    # Each vertical move enters a new tile.
    # Some of these vertical moves also move us closer to X2.
    # Specifically, in 2 vertical moves, we can change our X-tile index by 1 
    # without any additional horizontal moves by zig-zagging.
    # The distance is:
    # dy = abs(Y1 - Y2)
    # dx = abs(X1 - X2)
    # cost = dy + max(0, (2 * dx - dy + 1) // 2)
    # Let's check Sample 1: X1=2, Y1=0, X2=0, Y2=5.
    # dy = 5, dx = 2.
    # cost = 5 + max(0, (4 - 5 + 1) // 2) = 5 + 0 = 5. Correct.
    # Sample 2: X1=1, Y1=1, X2=1, Y2=1.
    # dy = 0, dx = 0.
    # cost = 0 + 0 = 0. Correct.
    
    # Final Formula:
    # X = (x - (y % 2)) // 2
    # Y = y
    # dx = abs(X1 - X2)
    # dy = abs(Y1 - Y2)
    # ans = dy + max(0, (2 * dx - dy + 1) // 2)
    
    # Implementation using a single print statement to avoid loops/assignments
    # (though assignments are allowed, I'll keep it clean).
    
    print((lambda sx, sy, tx, ty: (
        (lambda x1, y1, x2, y2: (
            (lambda dx, dy: dy + max(0, (2 * dx - dy + 1) // 2))(
                abs(x1 - x2), abs(y1 - y2)
            ))
        ))((sx - (sy % 2)) // 2, sy, (tx - (ty % 2)) // 2, ty)
    ))(sx, sy, tx, ty))

if __name__ == "__main__":
    solve()