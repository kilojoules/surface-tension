The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing expressions over statements. This encourages the use of high-level abstractions and prevents the manual management of counters or accumulators, which is often cleaner for mathematical problems. For this problem, I will use `abs()` for distances and `max()`/`min()` to implement the logic of the tiling pattern in a single expression.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling pattern:
    # When i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, if j is even, tiles are [0,1], [2,3]... (i=0, 2, 4...)
    # In row j, if j is odd, tiles are [-1,0], [1,2]... (i=1, 3, 5...)
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's analyze:
    # For a fixed j:
    # If j is even: i=0 (0+even=even) -> A_{0,j} & A_{1,j} are one tile.
    # If j is odd: i=1 (1+odd=even) -> A_{1,j} & A_{2,j} are one tile.
    
    # Let's transform coordinates to a grid of tiles.
    # Each tile can be identified by (tile_x, tile_y).
    # For a cell (i, j):
    # If (i + j) % 2 == 0: it's the left half of a tile. Tile ID: ( (i - (j%2)) // 2, j )
    # If (i + j) % 2 == 1: it's the right half of a tile. Tile ID: ( (i - (j%2)) // 2, j )
    # Actually, the tile index for cell (i, j) is:
    # tile_x = (i - (j % 2)) // 2
    # tile_y = j
    
    # However, the cost to move between tiles is the Manhattan distance in the tile-grid,
    # but moving vertically might be "cheaper" or "more expensive" depending on alignment.
    # Let's redefine:
    # A cell (i, j) belongs to tile T(i, j).
    # T(i, j) = ((i - (j % 2)) // 2, j)
    
    # The distance between (sx, sy) and (tx, ty) in this tile-grid:
    # dx = abs(tile_x_s - tile_x_t)
    # dy = abs(tile_y_s - tile_y_t)
    # The cost is dx + dy, but we must account for the fact that 
    # moving vertically might land us in a tile that already covers the x-range.
    
    # Let's use the coordinate transformation:
    # x' = i - (j % 2) / 2  <-- not quite.
    # Let's use: 
    # tile_x(i, j) = (i - (j % 2)) // 2
    # tile_y(i, j) = j
    
    # The distance is abs(tile_x_s - tile_x_t) + abs(tile_y_s - tile_y_t).
    # But wait, if we move vertically, we might change the tile_x.
    # If we are at (tile_x, tile_y) and move to (tile_x, tile_y + 1),
    # the new tile_x is (i - ((j+1)%2)) // 2.
    # This is a known problem type. The optimal cost is:
    # cost = max(abs(sx - tx), abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2) 
    # No, that's for different tiles.
    
    # Correct logic for this specific tiling:
    # The distance is abs(sy - ty) + max(0, (abs(sx - tx) - (abs(sy - ty) + 1) // 2) * 2)
    # Actually, the simplest form is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The cost is dy + max(0, dx - (dy // 2 + 1)) if we move optimally.
    # Let's re-evaluate:
    # Each vertical step covers 1 unit of dy and can potentially cover 1 unit of dx 
    # because the tile boundaries shift.
    # In 2 vertical steps, we can move 1 unit of dx for "free" (by switching tiles).
    # The cost is dy + max(0, (dx - (dy + 1) // 2 + 1) // 2 * 2) ... no.
    
    # Let's use the property:
    # Cost = abs(sy - ty) + max(0, abs(sx - tx) - (abs(sy - ty) // 2 + 1))
    # Wait, the sample 1: (5,0) to (2,5). dx=3, dy=5.
    # Cost = 5 + max(0, 3 - (5//2 + 1)) = 5 + max(0, 3-3) = 5. Correct.
    # Sample 2: (3,1) to (4,1). dx=1, dy=0.
    # Cost = 0 + max(0, 1 - (0//2 + 1)) = 0 + max(0, 0) = 0. Correct.
    
    # Let's double check the logic:
    # To move dy, we must pay dy. During these dy moves, we can shift our x-position
    # by 1 unit every 2 vertical steps (because the tiles shift).
    # Specifically, in dy steps, we can cover (dy // 2) units of dx for free, 
    # plus potentially one more if the parity is right.
    # The number of dx units covered "for free" by dy vertical steps is (dy + 1) // 2.
    # The remaining dx is dx - (dy + 1) // 2.
    # These remaining dx must be covered by horizontal moves.
    # Since each horizontal tile is 2 units wide, 2 units of dx cost 1 toll.
    # So remaining cost is ceil(remaining_dx / 2) * 2 ? No.
    # If we move horizontally, we pay 1 toll per 2 units.
    # But we can't move "half" a tile.
    # The cost to cover remaining dx is (remaining_dx + 1) // 2 * 2 if we are not aligned.
    # Actually, the simplest formula is:
    # cost = dy + max(0, dx - (dy + 1) // 2) * 2 / 2 ... no.
    # Let's use: cost = dy + max(0, (dx - (dy + 1) // 2 + 1) // 2 * 2)
    # Wait, if dx=1, dy=0, cost=0. (1-1+1)//2 * 2 = 0.
    # If dx=2, dy=0, cost=1. (2-1+1)//2 * 2 = 2? No.
    
    # Let's use the tile coordinate logic:
    # s_tx = (sx - (sy % 2)) // 2
    # t_tx = (tx - (ty % 2)) // 2
    # cost = abs(sy - ty) + abs(s_tx - t_tx)
    # Let's test Sample 1: sx=5, sy=0, tx=2, ty=5
    # s_tx = (5 - 0) // 2 = 2
    # t_tx = (2 - 1) // 2 = 0
    # cost = abs(0 - 5) + abs(2 - 0) = 5 + 2 = 7. Wrong.
    
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are one tile."
    # This means for a fixed j:
    # If j is even: (0,j)-(1,j), (2,j)-(3,j) ... are tiles.
    # If j is odd: (-1,j)-(0,j), (1,j)-(2,j) ... are tiles.
    # This is exactly what I wrote. Let's re-calculate Sample 1.
    # S=(5,0), T=(2,5). 
    # S is in tile ((5-0)//2, 0) = (2, 0).
    # T is in tile ((2-1)//2, 5) = (0, 5).
    # Wait, the sample says cost is 5. My formula gives 7.
    # The sample says: Move left 1 (toll 0), Up 1 (toll 1), Left 1 (toll 0), Up 3 (toll 3), Left 1 (toll 0), Up 1 (toll 1).
    # Total toll = 1 + 3 + 1 = 5.
    # In this path, he moved dx=3 and dy=5.
    # He paid only for the vertical moves!
    # This means he used the vertical moves to change his x-coordinate.
    # Every time he moves up, he enters a new tile. 
    # If he is at the boundary of two tiles, he can move horizontally for free.
    # In Sample 1, he is at x=5.5. Tile is [4, 6] at j=0.
    # He moves to x=4.5. Still in tile [4, 6]. Toll 0.
    # Then he moves up to j=1. Tile at j=1 is [5, 7] or [3, 5].
    # Since x=4.5, he enters tile [3, 5]. Toll 1.
    # Then he moves to x=3.5. Still in tile [3, 5]. Toll 0.
    # Then he moves up to j=4. He enters tiles at j=2, 3, 4. Toll 3.
    # Then he moves to x=2.5. Tile at j=4 is [2, 4] or [4, 6].
    # Since x=3.5, he is in [2, 4]. Toll 0.
    # Then he moves up to j=5. Tile at j=5 is [1, 3] or [3, 5].
    # Since x=2.5, he enters [1, 3]. Toll 1.
    # Total = 1 + 3 + 1 = 5.
    
    # Observation: He can change his x-coordinate by 1 for free every time he moves vertically,
    # because the tile boundaries shift.
    # So in dy vertical steps, he can cover dy units of dx for free.
    # The cost is simply max(dy, dx) if he can leverage the shift? 
    # No, because he can only shift x by 1 per vertical step.
    # Actually, the cost is simply max(abs(sy - ty), abs(sx - tx)) 
    # BUT he can only move x by 1 for each y step.
    # If dx > dy, he needs extra horizontal moves.
    # Each 2 units of dx cost 1 toll.
    # Cost = dy + max(0, (dx - dy + 1) // 2)
    # Let's test Sample 1: dx=3, dy=5. Cost = 5 + max(0, (3-5+1)//2) = 5 + 0 = 5. Correct.
    # Sample 2: dx=1, dy=0. Cost = 0 + max(0, (1-0+1)//2) = 1. 
    # Wait, Sample 2 output is 0. 
    # In Sample 2, S=(3,1), T=(4,1). dx=1, dy=0.
    # S is in tile ((3-1)//2, 1) = (1, 1).
    # T is in tile ((4-1)//2, 1) = (1, 1).
    # They are in the same tile! So cost is 0.
    
    # Let's use the tile