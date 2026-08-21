```python
import sys

def solve():
    # Read input and map to integers
    # Input format: Sx Sy Tx Ty
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiles are 2x1 rectangles.
    # A square (i, j) and (i+1, j) are in the same tile if i+j is even.
    # This means for a fixed j:
    # If j is even, tiles are {(0,j),(1,j)}, {(2,j),(3,j)}, ...
    # If j is odd, tiles are {(-1,j),(0,j)}, {(1,j),(2,j)}, ...
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # A tile can be identified by (X, Y) where:
    # Y = y
    # X = (x + (y % 2)) // 2
    
    # Starting tile coordinates:
    s_x_tile = (sx + (sy % 2)) // 2
    s_y_tile = sy
    
    # Target tile coordinates:
    t_x_tile = (tx + (ty % 2)) // 2
    t_y_tile = ty
    
    # The distance between two tiles (x1, y1) and (x2, y2) in this 
    # transformed grid is the Manhattan distance if we could move 
    # freely. However, the movement rules are specific.
    # Moving 'n' units in a direction:
    # 1. Vertical move: Changes y. Each unit change in y enters a new tile.
    # 2. Horizontal move: Changes x. 
    #    - If we stay in the same row y, we might cross multiple tiles.
    #    - But the problem says "Choose a direction... and a positive integer n".
    #    - A single horizontal move of n units can span multiple tiles, 
    #      but we only pay when we "enter" a tile.
    #    - Actually, the rule is: "Each time he enters a tile, he pays a toll of 1."
    #    - If he is already in a tile and moves within it, toll is 0.
    #    - If he moves from tile A to tile B, he pays 1.
    #    - A single move of n units in one direction can cross multiple tiles.
    #    - For example, moving right by n units: he enters every tile 
    #      along the path.
    
    # Wait, the sample 1 explanation says:
    # "Move left by 1. Pay a toll of 0." 
    # This happens if the move stays within the same tile.
    # "Move up by 1. Pay a toll of 1."
    # This happens when entering a new tile.
    
    # Let's re-evaluate:
    # To get from (sx, sy) to (tx, ty):
    # The cost is the number of tile boundaries crossed.
    # Vertical distance: |sy - ty| boundaries are always crossed.
    # Horizontal distance: 
    # In a single row y, tiles are blocks of 2.
    # The number of tiles crossed horizontally is the number of boundaries 
    # between x-blocks.
    # The boundary between tile X and X+1 occurs at x = 2*X + (1 if y%2==0 else 0).
    
    # The minimum toll is simply the Manhattan distance in the tile-coordinate system:
    # Cost = |s_x_tile - t_x_tile| + |s_y_tile - t_y_tile|
    # However, there is a catch: a single move can cover multiple tiles.
    # "Choose a direction... and a positive integer n. Move n units."
    # If he moves right by n, he enters all tiles in between.
    # So the cost is the number of tiles entered.
    # If he starts in tile A and ends in tile B, and they are the same, cost is 0.
    # If he moves to a different tile, he pays 1 for the new tile.
    # If he moves across 3 tiles, he pays 3.
    
    # Actually, the problem is simpler:
    # He wants to minimize the number of tiles entered.
    # He can move any distance n in one go.
    # If he moves from (sx, sy) to (tx, sy) horizontally, he enters 
    # all tiles between the start tile and end tile.
    # The number of tiles entered is |s_x_tile - t_x_tile|.
    # Then he moves vertically, entering |s_y_tile - t_y_tile| tiles.
    # Total = |s_x_tile - t_x_tile| + |s_y_tile - t_y_tile|.
    
    # Let's check Sample 1: (5,0) to (2,5)
    # s_x_tile = (5 + (0%2)) // 2 = 2
    # s_y_tile = 0
    # t_x_tile = (2 + (5%2)) // 2 = (2+1)//2 = 1
    # t_y_tile = 5
    # Cost = |2 - 1| + |0 - 5| = 1 + 5 = 6? 
    # Sample 1 output is 5. Why?
    # Because he can move diagonally by combining a horizontal and vertical move.
    # If he moves to a position where the current tile and the target tile 
    # share a boundary, he can enter the target tile in one move.
    
    # Correct logic for this specific tile layout:
    # The cost is |s_x_tile - t_x_tile| + |s_y_tile - t_y_tile|.
    # But he can "save" 1 toll if he can transition between tiles 
    # without entering an intermediate one.
    # Actually, the parity of the coordinates matters.
    # The distance is simply the L1 distance in the transformed grid,
    # but we must account for the fact that he starts INSIDE a tile.
    # The number of transitions is what matters.
    # If he is in tile (sx_t, sy_t) and moves to (tx_t, ty_t),
    # the number of boundaries crossed is |sx_t - tx_t| + |sy_t - ty_t|.
    
    # Let's re-calculate Sample 1:
    # s_x_tile = (5 + 0)//2 = 2
    # s_y_tile = 0
    # t_x_tile = (2 + 1)//2 = 1
    # t_y_tile = 5
    # |2-1| + |0-5| = 6. Still 6.
    # Wait, the sample says: "Move left by 1. Pay 0."
    # (5.5, 0.5) is in tile A_{5,0}. Since 5+0=5 (odd), A_{5,0} is paired with A_{4,0}.
    # So (5.5, 0.5) and (4.5, 0.5) are in the same tile.
    # Moving left by 1 unit takes him to (4.5, 0.5), still in the same tile. Toll = 0.
    # Now he is at (4.5, 0.5) in tile {A_{4,0}, A_{5,0}}.
    # He moves up to (4.5, 1.5). He enters tile A_{4,1} (since 4+1=5, it's paired with A_{3,1}).
    # Toll = 1.
    # Now he is in tile {A_{3,1}, A_{4,1}}.
    # He moves left to (3.5, 1.5). Still in the same tile. Toll = 0.
    # Then he moves up 3 units to (3.5, 4.5).
    # He enters tiles at y=2, 3, 4. Toll = 3.
    # Finally, he moves up 1 unit to (3.5, 5.5).
    # He enters tile A_{3,5} (paired with A_{2,5}). Toll = 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # The pattern is: he can change his X-tile index for free if he is in a 
    # 2x1 tile and moves to the other half of it.
    # This means he can effectively change his x from x to x-1 or x+1 
    # without changing the tile index X = (x + (y%2)) // 2.
    # The cost is simply |s_x_tile - t_x_tile| + |s_y_tile - t_y_tile|.
    # Let's re-calculate Sample 1 with this:
    # s_x_tile = (5 + (0%2)) // 2 = 5 // 2 = 2
    # t_x_tile = (2 + (5%2)) // 2 = 3 // 2 = 1
    # |2 - 1| + |0 - 5| = 1 + 5 = 6.
    # Still 6. Let me re-read.
    # A_{i,j} and A_{i+1,j} are same tile if i+j is even.
    # For j=0: A_{0,0}A_{1,0}, A_{2,0}A_{3,0}, A_{4,0}A_{5,0} ...
    # (5,0) is in A_{5,0}. i=5, j=0. i+j=5 (odd).
    # The rule says if i+j is even, A_{i,j} and A_{i+1,j} are same.
    # For i=4, j=0: 4+0=4 (even). So A_{4,0} and A_{5,0} are the same tile.
    # So (5.5, 0.5) is in the tile {A_{4,0}, A_{5,0}}.
    # For j=5: i+5 is even if i is odd.
    # i=1: 1+5=6 (even). So A_{1,5} and A_{2,5} are the same tile.
    # i=3: 3+5=8 (even). So A_{3,5} and A_{4,5} are the same tile.
    # (2.5, 5.5) is in A_{2,5}. Since 2+5=7 (odd), it's paired with A_{1,5} (1+5=6).
    # So (2.5, 5.5) is in tile {A_{1,5}, A_{2,5}}.
    
    # Let's use the formula X = (i + (j % 2)) // 2
    # For (5,0): X = (5 + 0) // 2 = 2.
    # For (2,5): X = (2 + 1) // 2 = 1.
    # Distance = |2 - 1| + |0 - 5| = 6.
    # Wait, the sample says 5. Let me re-read the sample path.
    # 1. (5.5, 0.5) -> (4.5, 0.5): Same tile {A_{4,0}, A_{5,0}}. Cost 0.
    # 2. (4.5, 0.5) -> (4.5, 1.5): Enters tile {A_{3,1}, A_{4,1}}. Cost 1.
    # 3. (4.5, 1.5) -> (3.5, 1.5): Same tile {A_{3,1}, A_{4,1}}. Cost 0.
    # 4. (3.5, 1.5) -> (3.5, 4.5): Enters tiles at y=2, 3, 4. Cost 3.
    # 5. (3.5, 4.5) -> (3.5, 5.5): Enters tile {A_{3,5}, A_{4,5}}. Cost 1.
    # Total = 5.
    # In this path, he reached (3.5, 5.5). But the target is (2.5, 5.5).
    # (3.5, 5.5) and (2.5, 5.5) are in DIFFERENT tiles.
    # {A_{3,5}, A_{4,5}} and {A_{1,5}, A_{2,5}}.
    # So he