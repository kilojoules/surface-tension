```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even, tiles are [0,1][2,3]... (horizontal pairs starting at even i)
    # If j is odd, tiles are [1,2][3,4]... (horizontal pairs starting at odd i)
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # For a cell (i, j), its tile ID can be represented as (u, v).
    # If j is even: tile is {(i, j), (i+1, j)} for even i. 
    #   u = i // 2, v = j // 2, type = 'even_row'
    # If j is odd: tile is {(i, j), (i+1, j)} for odd i.
    #   u = (i-1) // 2, v = (j-1) // 2, type = 'odd_row'
    
    # Actually, a simpler way to think about the distance:
    # The cost to move between two tiles is the L1 distance in the "tile grid".
    # Let's map (x, y) to tile coordinates (U, V).
    # For a cell (x, y):
    # If (x + y) is even, it's the left half of a tile.
    # If (x + y) is odd, it's the right half of a tile.
    # The tile containing (x, y) can be uniquely identified by:
    # V = y
    # U = (x + y) // 2 if (x + y) is even else (x + y - 1) // 2
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even, tiles are {0,1}, {2,3}, {4,5}...
    # If j is odd, tiles are {-1,0}, {1,2}, {3,4}...
    
    # Let's use the property: Tile ID U = (x + (y % 2)) // 2, V = y
    # The distance between (Ux, Vx) and (Ut, Vt) in this tiling is:
    # cost = |Vx - Vt| + max(0, |Ux - Ut| - (1 if (Vx-Vt)%2 == 0 else 0))
    # This is getting complex. Let's use the official logic for this specific problem:
    # The distance is |sy - ty| + max(0, abs(sx - tx) - (1 if (sy - ty) % 2 == 0 else 0))
    # But we must account for the parity of the starting tile.
    
    # Correct logic for this tiling:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # If we move vertically dy units, we cross dy tiles.
    # The horizontal distance dx is covered by tiles of width 2.
    # The number of horizontal tiles is ceil(dx / 2).
    # However, if we move vertically, we might "skip" a horizontal boundary.
    
    # The minimum cost is:
    # dy + max(0, (dx + 1) // 2 - (1 if dy % 2 == 0 and (sx + sy) % 2 == (tx + ty) % 2 else 0))
    # Actually, the simplest formula for this problem is:
    # cost = dy + max(0, (dx + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2 - (1 if dy % 2 == 0 else 0))
    # Let's refine:
    # If we are in the same tile, cost is 0.
    # Two cells (sx, sy) and (tx, ty) are in the same tile if:
    # sy == ty AND (sx + sy) % 2 == 0 AND tx == sx + 1
    # OR sy == ty AND (sx + sy) % 2 != 0 AND tx == sx - 1
    
    # The general formula for the distance between tiles in this specific layout is:
    # dist = abs(sy - ty) + max(0, (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0) + 1) // 2 - (1 if abs(sy-ty)%2 == 0 else 0))
    # Let's test Sample 1: 5 0, 2 5 -> dx=3, dy=5. (5+0)%2=1, (2+5)%2=1.
    # dist = 5 + max(0, (3 + 0 + 1)//2 - 0) = 5 + 2 = 7. Incorrect. Sample 1 is 5.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # Vertical cost is always |sy - ty|.
    # Horizontal cost: we move in blocks of 2.
    # The number of horizontal boundaries we MUST cross is:
    # If we change y, the "parity" of the tile boundaries shifts.
    # The formula is: cost = abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - (1 if abs(sy - ty) % 2 == 0 else 0))
    # Wait, if sy == ty and they are in the same tile, cost is 0.
    # Sample 2: 3 1, 4 1 -> dx=1, dy=0. (3+1)%2=0. A_{3,1} and A_{4,1} are same tile?
    # Rule: i+j even => A_{i,j} and A_{i+1,j} same tile.
    # i=3, j=1 => i+j=4 (even). So A_{3,1} and A_{4,1} are one tile. Cost 0.
    # Formula: 0 + max(0, (1+1)//2 - 1) = 0. Correct.
    # Sample 1: 5 0, 2 5 -> dx=3, dy=5.
    # Formula: 5 + max(0, (3+1)//2 - 0) = 5 + 2 = 7. Still 7.
    
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # If he moves from tile A to tile B, he pays 1.
    # This is simply the distance in the dual graph.
    # The tiles are nodes. Two nodes are connected if tiles share an edge.
    # Tile coordinates: V = y, U = (x + (y % 2)) // 2
    # Distance = |V1 - V2| + |U1 - U2|
    # Let's check Sample 1: (5, 0) and (2, 5)
    # S: V=0, U=(5 + 0)//2 = 2
    # T: V=5, U=(2 + 1)//2 = 1
    # Dist = |0 - 5| + |2 - 1| = 5 + 1 = 6. Still not 5.
    # Wait, the distance is |V1 - V2| + |U1 - U2|, but we can move diagonally?
    # No, the moves are strictly N, S, E, W.
    # But a single move can span multiple tiles.
    # "Choose a direction and a positive integer n. Move n units."
    # This means he can jump over tiles. But he pays for EACH tile he enters.
    # This is equivalent to the L1 distance in the tile grid.
    # Let's re-calculate Sample 1 with U = (x + (y % 2)) // 2, V = y:
    # S: x=5, y=0 => U = (5+0)//2 = 2, V = 0
    # T: x=2, y=5 => U = (2+1)//2 = 1, V = 5
    # Dist = |2-1| + |0-5| = 6.
    # Is there a mistake in my U calculation?
    # If y=0, tiles are {0,1}, {2,3}, {4,5}, {6,7}. x=5 is in tile {4,5}. Index U=2.
    # If y=5, tiles are {1,2}, {3,4}, {5,6}. x=2 is in tile {1,2}. Index U=0.
    # Wait, if y=5 (odd), tiles are A_{i,j} and A_{i+1,j} where i+j is even.
    # i+5 is even => i is odd. Tiles are {1,2}, {3,4}, {5,6}...
    # x=2 is in tile {1,2}. That is the 0-th tile (i=1).
    # So S: U=2, V=0; T: U=0, V=5. Dist = |2-0| + |0-5| = 7.
    # Still not 5. Let me re-read the sample explanation.
    # Sample 1: (5,0) to (2,5).
    # 1. Move left 1: (4,0). Same tile? x=5, y=0 (5+0=5 odd), x=4, y=0 (4+0=4 even).
    # A_{4,0} and A_{5,0} are one tile because 4+0 is even. So (5,0) and (4,0) are in the same tile. Toll 0.
    # 2. Move up 1: (4,1). New tile. Toll 1.
    # 3. Move left 1: (3,1). x=4, y=1 (4+1=5 odd), x=3, y=1 (3+1=4 even).
    # A_{3,1} and A_{4,1} are one tile because 3+1 is even. Toll 0.
    # 4. Move up 3: (3,4). Crosses y=2, 3, 4. Toll 3.
    # 5. Move left 1: (2,4). x=3, y=4 (3+4=7 odd), x=2, y=4 (2+4=6 even).
    # A_{2,4} and A_{3,4} are one tile because 2+4 is even. Toll 0.
    # 6. Move up 1: (2,5). New tile. Toll 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # Analysis: He can move horizontally for free if he is within the same 2x1 tile.
    # He can move vertically and pay 1 per unit.
    # The key is that he can change his "horizontal tile" for free by moving vertically.
    # In Sample 1, he used vertical moves to align himself with the target x.
    # The cost is simply |sy - ty| + (1 if he needs an extra horizontal move else 0).
    # But he can only move horizontally for free if he is in the correct parity.
    # Let's use the coordinate system:
    # A cell (x, y) belongs to tile ( (x + (y%2)) // 2, y )
    # The distance is |y1 - y2| + |(x1 + (y1%2))//2 - (x2 + (y2%2))//2|
    # Wait, the sample 1 result is 5. |0-5| + |(5+0)//2 - (2+1)//2| = 5 + |2 - 1| = 6.
    # Still 6. Let me re-calculate the tile index for T(2,5):
    # y=5 is odd. Tile is A_{i,j} and A_{i+1,j} where i+j is even.
    # i+5 is even => i is odd.
    # For x=2, the tile is A_{1,5} and A_{2,5}. (i=1).
    # The index U for i=1 is (1-1)//2 =