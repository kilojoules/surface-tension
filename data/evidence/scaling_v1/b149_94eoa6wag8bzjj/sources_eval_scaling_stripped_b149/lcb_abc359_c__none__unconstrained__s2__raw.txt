```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule:
    # If i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), etc. if j is even,
    # and pairs (1,2), (3,4), etc. if j is ODBC.
    # Essentially, a tile is defined by the pair (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a coordinate system where each unit 
    # represents a tile boundary.
    # For a fixed y, the tiles are blocks of 2 units wide.
    # The "tile index" in x for a given y is:
    # If y is even, tiles are [0,1], [2,3]... -> index is x // 2
    # If y is odd, tiles are [-1,0], [1,2]... -> index is (x+1) // 2
    
    # A simpler way to think about this:
    # The cost to move between two tiles is the distance in a graph.
    # The tiles form a grid where some edges have weight 0 and some have weight 1.
    # Moving within a tile is free. 
    # Moving from tile (tx, ty) to (tx, ty+1) always costs 1.
    # Moving from tile (tx, ty) to (tx+1, ty) costs 1, UNLESS they are the same tile.
    
    # Let's normalize the coordinates.
    # For any (x, y), the tile it belongs to can be identified by:
    # TileX = (x + (y % 2)) // 2
    # TileY = y
    
    # The distance between (sx, sy) and (tx, ty) in this tile-grid:
    # The vertical distance is always |sy - ty|.
    # The horizontal distance is |TileX_s - TileX_t|.
    # However, moving vertically might change the TileX of the current tile.
    
    # Let's use the property:
    # Cost = max(|sy - ty|, |TileX_s - TileX_t|) is NOT correct because 
    # vertical moves always cost 1 and horizontal moves cost 1 per tile boundary.
    # Actually, the cost is simply:
    # cost = abs(sy - ty) + abs(TileX_s - TileX_t)
    # But wait, if we move vertically, we might land in a tile that is 
    # horizontally closer to the target.
    
    # Correct logic:
    # Let x1, y1 be start and x2, y2 be target.
    # The cost is abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # But we can optimize by picking whether to move to y2 first or x2 first.
    # Actually, the distance is simply:
    # dist = abs(y1 - y2) + abs( (x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2 )
    # Let's check Sample 1: 5 0 -> 2 5
    # TileS = (5 + 0)//2 = 2, TileT = (2 + (5%2))//2 = (2+1)//2 = 1
    # Cost = abs(0 - 5) + abs(2 - 1) = 5 + 1 = 6. 
    # Sample 1 says 5. Why?
    # Because we can move to y=5 first, then the TileX might change.
    # At y=0, x=5 is TileX=2.
    # At y=5, x=5 is TileX=(5+1)//2 = 3.
    # Then moving from TileX=3 to TileX=1 costs 2. Total 5 + 2 = 7.
    # Or move x first: TileX 2 -> 1 costs 1. Then y 0 -> 5 costs 5. Total 6.
    # Wait, the sample says 5. Let's re-read.
    # "Move left by 1 (toll 0), Move up by 1 (toll 1)..."
    # If he is at (5.5, 0.5), he is in tile A_{5,0} and A_{6,0} (since 5+0 is odd, 
    # the rule says i+j even -> A_{i,j} and A_{i+1,j} are same. 
    # For i=4, j=0, 4+0=even, so A_{4,0} and A_{5,0} are one tile.
    # So (5.5, 0.5) is in the tile {A_{4,0}, A_{5,0}}.
    # Let's redefine: TileX = (x + (j % 2)) // 2 is wrong.
    # If i+j is even, i and i+1 are together.
    # If j=0: (0,1), (2,3), (4,5) -> x=5 is in tile index 2.
    # If j=1: (1,2), (3,4), (5,6) -> x=5 is in tile index 2.
    # The tile index for x at height j is:
    # If j is even: x // 2 if x is odd, or x // 2 if x is even? 
    # Let's use: TileX(x, j) = (x + (j % 2)) // 2
    # Sample 1: S(5,0), T(2,5)
    # TileX(5, 0) = (5 + 0) // 2 = 2
    # TileX(2, 5) = (2 + 1) // 2 = 1
    # The cost is abs(y1 - y2) + abs(TileX(x1, y1) - TileX(x2, y2))
    # But we can change the TileX by moving vertically.
    # The cost is min(
    #    abs(y1 - y2) + abs(TileX(x1, y1) - TileX(x2, y2)),
    #    abs(y1 - y2) + abs(TileX(x1, y2) - TileX(x2, y2)) 
    # ) 
    # Wait, TileX(x1, y2) is the tile x1 belongs to at height y2.
    # Let's use the property: the distance is abs(y1 - y2) + abs(TileX(x1, y1) - TileX(x2, y2))
    # but we can "shift" the starting TileX by 1 if we move vertically.
    # Actually, the simplest formula is:
    # ans = abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # Let's re-calculate Sample 1: abs(0-5) + abs((5+0)//2 - (2+1)//2) = 5 + abs(2 - 1) = 6.
    # Still 6. Let's look at the sample path:
    # (5.5, 0.5) -> left 1 -> (4.5, 0.5). 
    # A_{5,0} and A_{4,0} are the same tile (4+0 is even). Toll 0.
    # (4.5, 0.5) -> up 1 -> (4.5, 1.5). 
    # A_{4,1} is a new tile. Toll 1.
    # (4.5, 1.5) -> left 1 -> (3.5, 1.5).
    # A_{4,1} and A_{3,1} are the same tile (3+1 is even). Toll 0.
    # This means we can change our TileX for free by moving vertically!
    # If we are in TileX=k at height j, and we move to height j+1,
    # we are now in a tile that covers either (2k, 2k+1) or (2k-1, 2k).
    # The cost is simply abs(y1 - y2) + abs(TileX(x1, y1) - TileX(x2, y2))
    # BUT, we can choose to start from either TileX(x1, y1) or TileX(x1, y1+1).
    # The distance is abs(y1 - y2) + abs(TileX(x1, y1) - TileX(x2, y2))
    # However, if we move vertically, we can effectively change our TileX by 1.
    # The correct formula is:
    # ans = abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # But we can subtract 1 if we can "cheat" the boundary.
    # Let's use: ans = abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # If we move vertically, we can reach TileX(x1, y1) or TileX(x1, y1+1).
    # The distance is abs(y1 - y2) + min(abs(TileX(x1, y1) - TileX(x2, y2)), 
    #                                  abs(TileX(x1, y1+1) - TileX(x2, y2)))
    # Wait, if y1 == y2, the cost is just abs(TileX(x1, y1) - TileX(x2, y2)).
    # If y1 != y2, we can potentially reduce the TileX distance by 1.
    # Let's check Sample 1: abs(0-5) + min(abs(2-1), abs(TileX(5, 1) - 1))
    # TileX(5, 1) = (5+1)//2 = 3. min(1, abs(3-1)) = 1. Still 6.
    # Let's re-read: "Move left by 1. Pay a toll of 0."
    # (5.5, 0.5) is in A_{5,0}. A_{4,0} and A_{5,0} are one tile.
    # So moving to (4.5, 0.5) is free.
    # Then move up to (4.5, 1.5). A_{4,1} is a new tile. Toll 1.
    # Then move left to (3.5, 1.5). A_{3,1} and A_{4,1} are one tile. Toll 0.
    # This means at y=1, we are in the tile covering x=3 and x=4.
    # This is TileX = (3+1)//2 = 2.
    # So the cost is abs(y1 - y2) + abs(TileX(x_adjusted, y2) - TileX(x2, y2))
    # where x_adjusted is x1 shifted by at most 1 to enter the best tile.
    # Actually, the most direct formula is:
    # ans = abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # But if y1 != y2, we can potentially reduce this by 1 if the parity allows.
    # Let's use the logic: 
    # The distance is abs(y1 - y2) + abs(TileX(x1, y1) - TileX(x2, y2))
    # But we can move to an adjacent tile for free if we move vertically.
    # The correct answer is abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # MINUS 1 if (y1 != y2) AND (some condition).
    # Let's try: ans = abs(y1 - y2) + abs((x1 + (y1 % 2)) // 2 - (x2 + (y2 % 2)) // 2)
    # If y1