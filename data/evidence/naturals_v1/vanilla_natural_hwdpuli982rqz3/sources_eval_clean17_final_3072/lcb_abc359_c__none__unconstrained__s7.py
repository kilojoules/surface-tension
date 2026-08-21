import sys

def solve():
    # Read input from stdin
    # The input consists of Sx, Sy, Tx, Ty
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are horizontal 2x1 blocks.
    # In row j, the boundaries between tiles are at x = k where (k + j) is odd.
    # To move from (sx, sy) to (tx, ty), we want to minimize the number of tiles entered.
    
    # Let's transform the coordinates to a system where the grid is uniform.
    # Consider the "dual" grid of tiles.
    # A tile can be identified by (i, j) where i is the left index of the 2x1 block.
    # If (i + j) is even, the tile is {i, i+1} x {j}.
    # If (i + j) is odd, the tile is {i} x {j}. Wait, the rule says:
    # "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # This means for a fixed j:
    # If j is even: A_{0,j}A_{1,j}, A_{2,j}A_{3,j}, ... are tiles.
    # If j is odd: A_{1,j}A_{2,j}, A_{3,j}A_{4,j}, ... are tiles.
    
    # Let's map each square (x, y) to a tile coordinate (u, v).
    # v = y.
    # If y is even, x is in tile u = x // 2.
    # If y is odd, x is in tile u = (x - 1) // 2 if x > 0 else -1.
    # More simply: u = (x + (y % 2)) // 2.
    
    # Let f(x, y) = ((x + (y % 2)) // 2, y)
    # The distance between two tiles (u1, v1) and (u2, v2) in this specific 
    # brick-wall layout is known to be:
    # cost = max(|u1 - u2|, (|v1 - v2| + 1) // 2) 
    # However, the most reliable formula for this specific problem (which is a 
    # known competitive programming problem) is:
    # Let dx = tx - sx, dy = ty - sy.
    # The minimum toll is max(abs(dx), abs(dy), (abs(dx) + abs(dy) + 1) // 2) 
    # is for a different grid.
    
    # For this specific brick pattern:
    # The distance is max(|u1 - u2|, abs(v1 - v2)) where we can move 
    # diagonally in the (u, v) space.
    # Let's use the coordinate transformation:
    # u = x + y
    # v = x - y
    # This is not quite right for bricks.
    
    # Correct logic for this brick pattern:
    # To move from (sx, sy) to (tx, ty):
    # The number of vertical boundaries crossed is abs(sy - ty).
    # The number of horizontal boundaries crossed is tricky.
    # Let's use the property:
    # Min Toll = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Wait, that's for a different grid.
    
    # Let's re-evaluate:
    # In row j, boundaries are at x = k where k % 2 != j % 2.
    # To get from (sx, sy) to (tx, ty), we can move to a "highway" (a tile boundary).
    # The minimum cost is actually:
    # cost = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # No, the sample 1: (5,0) to (2,5). dx=3, dy=5. 
    # max(3, 5, (3+5+1)//2) = 5. Correct.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0.
    # max(1, 0, (1+0+1)//2) = 1. But sample output is 0.
    # Why? Because (3,1) and (4,1) are in the same tile.
    # i=3, j=1 => i+j=4 (even). So A_{3,1} and A_{4,1} are one tile.
    
    # Let's use the tile coordinates:
    # u = (x + (y % 2)) // 2
    # v = y
    # The distance between (u1, v1) and (u2, v2) is:
    # We can move from (u, v) to (u, v +/- 1) with cost 1.
    # We can move from (u, v) to (u +/- 1, v) with cost 1.
    # But we can also move from (u, v) to (u, v +/- 1) and (u +/- 1, v +/- 1) 
    # because of the offset.
    # Specifically, the tile (u, v) shares a boundary with:
    # (u, v-1), (u, v+1), (u-1, v), (u+1, v)
    # AND (u-1, v-1), (u+1, v-1), (u-1, v+1), (u+1, v+1) depending on parity.
    
    # The distance is actually:
    # dist = max(abs(u1 - u2), abs(v1 - v2), (abs(u1 - u2) + abs(v1 - v2) + 1) // 2)
    # Wait, the simplest form for this specific grid is:
    # Let dx = tx - sx, dy = ty - sy.
    # The answer is max(abs(dx), abs(dy), (abs(dx) + abs(dy) + 1) // 2) 
    # ONLY if we are moving between centers of squares.
    # But we must account for the fact that we start inside a tile.
    
    # Let's use the (u, v) coordinates:
    # u = (x + (y % 2)) // 2
    # v = y
    # The distance between (u1, v1) and (u2, v2) is:
    # cost = max(abs(u1 - u2), abs(v1 - v2))
    # Let's check Sample 1: (5,0) -> u=(5+0)//2=2, v=0; (2,5) -> u=(2+1)//2=1, v=5.
    # max(|2-1|, |0-5|) = 5. Correct.
    # Sample 2: (3,1) -> u=(3+1)//2=2, v=1; (4,1) -> u=(4+1)//2=2, v=1.
    # max(0, 0) = 0. Correct.
    # Sample 3: (2552608206527595, 5411232866732612) -> u=1276304103263797, v=5411232866732612
    # (771856005518028, 7206210729152763) -> u=385928002759014, v=7206210729152763
    # max(|1276304103263797 - 385928002759014|, |5411232866732612 - 7206210729152763|)
    # max(890376100504783, 1794977862420151) = 1794977862420151. Correct.

    u1 = (sx + (sy % 2)) // 2
    v1 = sy
    u2 = (tx + (ty % 2)) // 2
    v2 = ty
    
    print(max(abs(u1 - u2), abs(v1 - v2)))

if __name__ == "__main__":
    solve()