The constraint to avoid explicit loops and recursion forces a functional programming style, encouraging the use of declarative expressions over imperative state updates. This approach often leads to more concise logic for mathematical problems, as it leverages Python's powerful built-in sequence processing and conditional expressions. To solve this problem, I will use `abs()` for distances and nested `if-else` expressions (ternary operators) to calculate the cost based on the parity of the coordinates and the relative positions of the start and target points.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are paired (0,1), (2,3)... if j is even
    # and (-1,0), (1,2)... if j is odd.
    
    # Let's define a coordinate transformation to a grid of tiles.
    # Each tile can be identified by (tile_x, tile_y).
    # For a square (i, j):
    # If (i + j) is even, it's the left half of a tile.
    # If (i + j) is odd, it's the right half of a tile.
    
    # A square (i, j) belongs to tile:
    # tile_y = j
    # tile_x = (i if (i + j) % 2 == 0 else i - 1) // 2
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's re-evaluate:
    # For a fixed j:
    # If j is even: (0,j) & (1,j) are tile 0, (2,j) & (3,j) are tile 1...
    # If j is odd: (-1,j) & (0,j) are tile 0, (1,j) & (2,j) are tile 1...
    
    # More simply: Square (i, j) belongs to tile ( (i + (j % 2)) // 2, j )
    
    # Let's define the tile coordinates for S and T
    # s_tile_x = (sx + (sy % 2)) // 2
    # s_tile_y = sy
    # t_tile_x = (tx + (ty % 2)) // 2
    # t_tile_y = ty
    
    # The cost to move between tiles:
    # Moving vertically (changing tile_y) always costs 1 per unit.
    # Moving horizontally (changing tile_x) costs 1 per unit.
    # However, we can move horizontally "for free" if we are within the same tile.
    # The distance in tile_x is abs(s_tile_x - t_tile_x).
    # The distance in tile_y is abs(s_tile_y - t_tile_y).
    
    # The total cost is the Manhattan distance in the tile-grid:
    # cost = abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y)
    # But we must check if we start and end in the same tile.
    # If we are in the same tile, the cost is 0.
    # The Manhattan distance formula already covers this.
    
    # Let's double check Sample 1: S(5,0), T(2,5)
    # s_tile_x = (5 + (0%2)) // 2 = 5 // 2 = 2
    # s_tile_y = 0
    # t_tile_x = (2 + (5%2)) // 2 = (2 + 1) // 2 = 1
    # t_tile_y = 5
    # cost = abs(2 - 1) + abs(0 - 5) = 1 + 5 = 6? 
    # Sample 1 says 5. Let's re-read.
    
    # "Each time he enters a tile, he pays a toll of 1."
    # This means the starting tile is free.
    # The cost is the number of tile boundaries crossed.
    # In a grid, the number of boundaries crossed is the Manhattan distance.
    # Wait, the sample says 5. My calculation gave 6.
    # Let's re-examine the tile boundaries.
    # S(5,0) is in tile ((5+0)//2, 0) = (2, 0).
    # T(2,5) is in tile ((2+1)//2, 5) = (1, 5).
    # Manhattan distance is |2-1| + |0-5| = 6.
    # Why is it 5? 
    # Because he can move to a tile that is adjacent diagonally? 
    # No, he moves in straight lines.
    # But he can move to (2, 0) -> (2, 5) -> (1, 5).
    # The tiles are 2x1. 
    # At y=0, tile is (2,0). At y=1, tile is ( (i+1)//2, 1).
    # If he is at x=5.5, y=0.5, he is in tile (2,0).
    # He moves to x=5.5, y=5.5. 
    # He passes through y=1, 2, 3, 4, 5.
    # The tiles he enters are:
    # y=1: tile((5+1)//2, 1) = (3, 1)
    # y=2: tile((5+0)//2, 2) = (2, 2)
    # y=3: tile((5+1)//2, 3) = (3, 3)
    # y=4: tile((5+0)//2, 4) = (2, 4)
    # y=5: tile((5+1)//2, 5) = (3, 5)
    # Then he moves to x=2.5, y=5.5.
    # He is already in tile (3, 5). He moves to tile (1, 5).
    # He enters tile (2, 5) and then tile (1, 5).
    # Total tiles entered: 5 (vertical) + 2 (horizontal) = 7? Still not 5.
    
    # Let's reconsider: he can choose n.
    # He can move from (5.5, 0.5) to (5.5, 5.5) in one move.
    # He enters all tiles along the segment.
    # The number of tiles he enters is the number of distinct tiles the segment intersects.
    # For a vertical segment at x, the tiles are ((x+ (y%2))//2, y).
    # As y changes, the tile index changes.
    # For a fixed x, the tile index is (x+0)//2 for even y and (x+1)//2 for odd y.
    # These two indices are either the same or differ by 1.
    # Specifically, if x is odd, (x+0)//2 == (x+1)//2 - 1.
    # If x is even, (x+0)//2 == (x+1)//2.
    
    # If x is even, the vertical segment stays in the same "column" of tiles.
    # If x is odd, it toggles between two columns.
    
    # Let's use the property:
    # Cost = abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y)
    # But if we can pick an x or y that minimizes the transitions.
    # For a vertical move at x, the cost is the number of distinct tiles.
    # If x is even, the tile index is always x//2. Cost is just dy.
    # If x is odd, the tile index alternates. Cost is dy.
    # Wait, the only way to reduce cost is to move to an x where the vertical 
    # movement is "cheaper" or a y where horizontal is "cheaper".
    # But vertical movement always crosses dy boundaries.
    # The only "free" move is moving within a 2x1 tile (horizontal move).
    
    # Correct logic:
    # The distance is abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y).
    # However, we can move to an adjacent tile diagonally in 1 move?
    # No, but we can move to a coordinate x' such that the vertical path is cheaper.
    # Actually, the distance is simply:
    # abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y)
    # Let's re-check Sample 1: S(5,0), T(2,5)
    # s_tile_x = (5 + (0%2)) // 2 = 2
    # t_tile_x = (2 + (5%2)) // 2 = 1
    # cost = |2-1| + |0-5| = 6. Still 6.
    # Wait, the sample says: "Move left by 1 (toll 0), Up by 1 (toll 1), Left by 1 (toll 0)..."
    # S(5,0) -> (4,0). S is in tile ((5+0)//2, 0) = (2,0). (4,0) is in tile ((4+0)//2, 0) = (2,0).
    # Toll 0. Then (4,0) -> (4,1). Tile is ((4+1)//2, 1) = (2,1). Toll 1.
    # Then (4,1) -> (3,1). Tile is ((3+1)//2, 1) = (2,1). Toll 0.
    # This means if we are at an x such that (x+y)%2 == 0, we can move x -> x-1 for free.
    # This is exactly what the tile definition says!
    # The cost is simply the Manhattan distance in the tile-grid, 
    # but we can move diagonally in the tile-grid for the cost of 1?
    # No, the sample shows: (2,0) -> (2,1) -> (2,1) -> (2,2) ...
    # The cost is max(abs(s_tile_x - t_tile_x), abs(s_tile_y - t_tile_y))? 
    # No, that's for 8-connectivity.
    # Let's look at the sample again.
    # (5,0) [Tile 2,0] -> (4,0) [Tile 2,0] -> (4,1) [Tile 2,1] -> (3,1) [Tile 2,1] -> (3,4) [Tiles 2,2 2,3 2,4] -> (2,4) [Tile 1,4] -> (2,5) [Tile 1,5]
    # Total cost: 1 (for y=1) + 3 (for y=2,3,4) + 1 (for y=5) = 5.
    # The x-coordinate was adjusted to stay in the same tile-column.
    # If we are at x, and we move to x', the cost is 0 if they are in the same tile.
    # This means we can change x to x-1 or x+1 for free if (x+y)%2 == 0.
    # This allows us to effectively move "diagonally" in the tile grid.
    # The cost is max(abs(s_tile_x - t_tile_x), abs(s_tile_y - t_tile_y)).
    # Let's check Sample 1: max(|2-1|, |0-5|) = 5. Correct!
    # Sample 2: S(3,1), T(4,1). 
    # s_tile_x = (3+1)//2 = 2, s_tile_y = 1.
    # t_tile_x = (4+1)//2 = 2, t_tile_y = 1.
    # max(|2-2|, |1-1|) = 0. Correct!
    
    # Final formula:
    # s_tx = (sx + (sy % 2)) // 2
    # t_tx = (tx + (ty % 2)) // 2
    # result = max(abs(