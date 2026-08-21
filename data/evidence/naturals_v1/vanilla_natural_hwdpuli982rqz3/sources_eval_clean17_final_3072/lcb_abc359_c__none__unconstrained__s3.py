import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is defined by:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the x-boundaries of tiles are at:
    # j even: ..., -2, 0, 2, 4, ...
    # j odd:  ..., -1, 1, 3, 5, ...
    # In general, the vertical boundaries between tiles in row j are x = k where k % 2 != j % 2.
    
    # Let's transform the coordinates to a system where the grid is uniform.
    # Consider the transformation: 
    # u = x + y
    # v = x - y
    # However, a simpler way is to look at the Manhattan distance and the 
    # specific structure of the "toll-free" moves.
    # A move is free if it stays within the same tile.
    # Horizontal move (x, y) -> (x+n, y): free if no boundary x = k (k%2 != y%2) is crossed.
    # Vertical move (x, y) -> (x, y+n): always crosses a boundary unless n=0.
    
    # Let's redefine the coordinates to make the tiles 1x1 blocks in a skewed grid.
    # Let X = x, Y = y.
    # The boundary between tiles in row Y is at X = k where k % 2 != Y % 2.
    # This is equivalent to saying the boundary is at X + Y = odd.
    # Let u = x + y and v = x - y.
    # A vertical move (0, 1) changes u by 1 and v by -1.
    # A horizontal move (1, 0) changes u by 1 and v by 1.
    
    # Actually, there is a known result for this specific tiling problem:
    # The distance is max(|sx - tx|, |sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # But let's derive it via the u, v coordinates.
    # Let u = x + y and v = x - y.
    # A tile consists of {(x, y), (x+1, y)} where x+y is even.
    # In (u, v) space:
    # Tile is {(u, v), (u+1, v+1)} where u is even.
    # Moving 'up' (0, 1): (u, v) -> (u+1, v-1). This always enters a new tile.
    # Moving 'right' (1, 0): (u, v) -> (u+1, v+1). This is free if u is even.
    
    # Let's use the property:
    # To move from (sx, sy) to (tx, ty), the minimum cost is:
    # cost = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Wait, the sample 1: (5,0) to (2,5). dx=3, dy=5. max(3, 5, (3+5+1)//2) = 5. Correct.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0. max(1, 0, (1+0+1)//2) = 1. 
    # But Sample 2 output is 0. Why?
    # (3,1): 3+1=4 (even). A_{3,1} and A_{4,1} are the same tile.
    # So moving from 3.5 to 4.5 in x is free.
    
    # Correct Logic:
    # Let dx = tx - sx, dy = ty - sy.
    # The cost is 0 if we can reach the target using only "free" horizontal moves.
    # A horizontal move at height y is free if we don't cross x = k where k%2 != y%2.
    # This means we can move freely between x and x+1 if x%2 == y%2.
    
    # Let's use the coordinate transformation:
    # x' = x, y' = y
    # The cost to move from (sx, sy) to (tx, ty) is:
    # f(sx, sy, tx, ty) = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # But we must account for the starting tile.
    # If we move horizontally first, we might save 1.
    # Let's evaluate the 4 possible first moves (or no move).
    # Actually, the formula is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty).
    # The minimum cost is (dx + dy + 1) // 2, unless we can utilize the 2x1 tiles.
    # The 2x1 tiles are horizontal.
    # If we are at (sx, sy) and we move to (sx+1, sy) or (sx-1, sy), it's free if sx+sy is even (for +1) or sx-1+sy is even (for -1).
    
    # Let's use the property:
    # Min cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # But we can potentially reduce sx or tx by 1 for free if the parity matches.
    # If (sx + sy) is even, we can move to (sx + 1, sy) for free.
    # If (sx + sy) is odd, we can move to (sx - 1, sy) for free.
    # This means we can effectively change sx to sx' where sx' is the nearest integer such that 
    # the move was free, and similarly for tx.
    
    # Let's refine:
    # We can reach any x in {sx, sx+1} if sx+sy is even, or {sx, sx-1} if sx+sy is odd, for free.
    # Let S_set = {sx} if sx+sy is odd else {sx, sx+1} ... no.
    # If (sx + sy) % 2 == 0, the tile is { (sx, sy), (sx+1, sy) }.
    # If (sx + sy) % 2 == 1, the tile is { (sx-1, sy), (sx, sy) }.
    
    # Let's define the range of x covered by the tile at (sx, sy) as [L1, R1]
    # and the range of x covered by the tile at (tx, ty) as [L2, R2].
    # L1 = sx if (sx + sy) % 2 == 1 else sx
    # R1 = sx + 1 if (sx + sy) % 2 == 0 else sx
    # Wait:
    # If (i+j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # For a point (x+0.5, y+0.5), i=x, j=y.
    # If (x+y) is even, the tile is {x, x+1} x {y, y+1}.
    # If (x+y) is odd, the tile is {x-1, x} x {y, y+1}.
    
    # Let x_left(x, y) and x_right(x, y) be the boundaries of the tile containing (x+0.5, y+0.5).
    # If (x+y) % 2 == 0: x_left = x, x_right = x + 1
    # If (x+y) % 2 == 1: x_left = x - 1, x_right = x
    
    # The distance between two tiles (L1, R1, y1) and (L2, R2, y2) is:
    # dx = max(0, L2 - R1, L1 - R2)
    # dy = abs(y1 - y2)
    # cost = max(dy, (dx + dy + 1) // 2)
    
    # However, we can pick ANY x1 in [L1, R1] and x2 in [L2, R2] (integer boundaries).
    # But the formula max(dy, (dx + dy + 1) // 2) is for the distance between points.
    # The minimum cost to get from tile 1 to tile 2 is:
    # min( max(abs(y1-y2), (abs(x1-x2) + abs(y1-y2) + 1)//2) ) 
    # where x1 is the boundary of tile 1 and x2 is the boundary of tile 2.
    # Actually, the simplest way:
    # The cost is max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # But we can shift sx to sx+1 if (sx+sy)%2==0, or to sx-1 if (sx+sy)%2==1.
    # And tx to tx+1 if (tx+ty)%2==0, or to tx-1 if (tx+ty)%2==1.
    
    # Let's test all 4 combinations of (sx', tx') where sx' is in {sx, sx+1 if (sx+sy)%2==0 else sx-1}
    # and tx' is in {tx, tx+1 if (tx+ty)%2==0 else tx-1}.
    
    # Correction: if (sx+sy)%2 == 0, the tile is [sx, sx+1]. Any x in [sx, sx+1] is "free".
    # To minimize max(dy, (dx+dy+1)//2), we want to minimize dx = abs(x1 - x2).
    # x1 can be any value in [sx, sx+1] if (sx+sy)%2==0 else [sx-1, sx].
    # x2 can be any value in [tx, tx+1] if (tx+ty)%2==0 else [tx-1, tx].
    
    # Let interval 1 be I1 and interval 2 be I2.
    # dx = max(0, max(I1[0], I2[0]) - min(I1[1], I2[1]))
    # cost = max(abs(sy - ty), (dx + abs(sy - ty) + 1) // 2)
    
    i1 = (sx, sx + 1) if (sx + sy) % 2 == 0 else (sx - 1, sx)
    i2 = (tx, tx + 1) if (tx + ty) % 2 == 0 else (tx - 1, tx)
    
    dx = max(0, max(i1[0], i2[0]) - min(i1[1], i2[1]))
    dy = abs(sy - ty)
    
    print(max(dy, (dx + dy + 1) // 2))

if __name__ == "__main__":
    solve()