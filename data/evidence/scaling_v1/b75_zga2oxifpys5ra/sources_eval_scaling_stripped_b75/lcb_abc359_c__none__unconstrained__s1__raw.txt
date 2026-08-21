```python
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
    # This means for a fixed j, the tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (1,2), (3,4), etc. if j is odd.
    # Essentially, a tile is defined by the pair (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a coordinate system where each unit 
    # movement in the new system corresponds to crossing a tile boundary.
    # The "tile-index" for a cell (i, j) is:
    # X_tile = (i + (j % 2)) // 2
    # Y_tile = j
    
    # However, the cost is based on entering a new tile.
    # Moving vertically always enters a new tile (cost 1 per unit).
    # Moving horizontally may or may not enter a new tile.
    # If we are at (sx, sy) and move to (tx, ty):
    # The vertical distance is |sy - ty|. Each step costs 1.
    # The horizontal distance is |sx - tx|. 
    # But we can optimize by picking the best y-level to travel horizontally.
    
    # Let's analyze the cost:
    # To move from (sx, sy) to (tx, ty), the total cost is:
    # cost = |sy - ty| + (cost to move from sx to tx at some height y)
    # At a specific height y, the cost to move from sx to tx is:
    # The number of tile boundaries crossed.
    # A boundary exists between i and i+1 if (i + y) is odd.
    # The number of boundaries between sx and tx at height y is:
    # floor((tx + (y%2))/2) - floor((sx + (y%2))/2) if tx > sx
    # Otherwise, floor((sx + (y%2))/2) - floor((tx + (y%2))/2)
    
    # Let f(i, y) = (i + (y % 2)) // 2
    # Cost = |sy - ty| + min( |f(tx, y) - f(sx, y)| ) for y in {sy, ty}
    # Actually, the path can be: 
    # 1. Move from sy to ty (cost |sy - ty|), then move from sx to tx at height ty.
    # 2. Move from sx to tx at height sy, then move from sy to ty (cost |sy - ty|).
    # 3. Move to some other height y, but that would only increase |sy - y| + |y - ty|.
    
    # The cost to move horizontally at height y is |f(tx, y) - f(sx, y)|.
    # Total cost = |sy - ty| + min(|f(tx, sy) - f(sx, sy)|, |f(tx, ty) - f(sx, ty)|)
    
    # Wait, there's a slight nuance: if we move vertically and land in the same 
    # tile as the destination, we might save 1. But the problem says "each time 
    # he enters a tile". Starting tile is free.
    
    # Let's use the formula:
    # cost = abs(sy - ty) + min(
    #     abs((tx + (sy % 2)) // 2 - (sx + (sy % 2)) // 2),
    #     abs((tx + (ty % 2)) // 2 - (sx + (ty % 2)) // 2)
    # )
    
    # Example 1: 5 0 -> 2 5
    # sy=0, ty=5. |0-5| = 5.
    # f(2, 0) = (2+0)//2 = 1; f(5, 0) = (5+0)//2 = 2. Diff = 1.
    # f(2, 5) = (2+1)//2 = 1; f(5, 5) = (5+1)//2 = 3. Diff = 2.
    # Total = 5 + min(1, 2) = 6. 
    # Wait, Sample 1 says 5. Let's re-read.
    # "Move left by 1. Pay 0." (5,0) to (4,0). (5+0)//2 = 2, (4+0)//2 = 2. Same tile.
    # "Move up by 1. Pay 1." (4,0) to (4,1). New tile.
    # "Move left by 1. Pay 0." (4,1) to (3,1). (4+1)//2 = 2, (3+1)//2 = 2. Same tile.
    # This means the cost is simply the distance in the transformed coordinate system.
    # The distance between (sx, sy) and (tx, ty) in this grid is:
    # dist = |sy - ty| + |f(tx, y_final) - f(sx, y_start)| ... no.
    
    # Correct logic:
    # The cost is |sy - ty| + |f(tx, ty) - f(sx, sy)| is not quite right because
    # the x-coordinate of the tile changes based on the parity of y.
    # Let's use the property:
    # Cost = abs(sy - ty) + abs((tx + (ty % 2)) // 2 - (sx + (sy % 2)) // 2)
    # Let's check Sample 1: abs(0-5) + abs((2+1)//2 - (5+0)//2) = 5 + abs(1 - 2) = 6. Still 6.
    # Let's re-read: "Move left by 1 (5,0)->(4,0) [0], Up 1 (4,0)->(4,1) [1], Left 1 (4,1)->(3,1) [0], Up 3 (3,1)->(3,4) [3], Left 1 (3,4)->(2,4) [0], Up 1 (2,4)->(2,5) [1]."
    # Total = 0+1+0+3+0+1 = 5.
    # In this path, he moved from (5,0) to (2,5).
    # The x-coordinates of the tiles were:
    # (5,0): tile (5+0)//2 = 2
    # (4,0): tile (4+0)//2 = 2
    # (4,1): tile (4+1)//2 = 2
    # (3,1): tile (3+1)//2 = 2
    # (3,4): tile (3+0)//2 = 1
    # (2,4): tile (2+0)//2 = 1
    # (2,5): tile (2+1)//2 = 1
    # The tiles visited were: Tile(2,0) -> Tile(2,1) -> Tile(2,2) -> Tile(2,3) -> Tile(2,4) -> Tile(1,4) -> Tile(1,5)
    # Wait, the movement (3,4) to (2,4) is within the same tile because (3+0)//2 = 1 and (2+0)//2 = 1.
    # The sequence of tiles:
    # Start: Tile((5+0)//2, 0) = (2, 0)
    # Move to (4,0): Tile((4+0)//2, 0) = (2, 0) - Cost 0
    # Move to (4,1): Tile((4+1)//2, 1) = (2, 1) - Cost 1
    # Move to (3,1): Tile((3+1)//2, 1) = (2, 1) - Cost 0
    # Move to (3,4): Tile((3+0)//2, 4) = (1, 4) - Cost 3 (enters tiles at y=2,3,4)
    # Move to (2,4): Tile((2+0)//2, 4) = (1, 4) - Cost 0
    # Move to (2,5): Tile((2+1)//2, 5) = (1, 5) - Cost 1
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # The general formula for the distance between (sx, sy) and (tx, ty) is:
    # cost = abs(sy - ty) + abs((tx + (ty % 2)) // 2 - (sx + (sy % 2)) // 2)
    # Let's re-calculate Sample 1: abs(0-5) + abs((2+1)//2 - (5+0)//2) = 5 + abs(1 - 2) = 6.
    # Still 6. What is wrong?
    # Let's look at the tile indices again:
    # Start: (sx, sy) -> Tile((sx + (sy%2))//2, sy)
    # End: (tx, ty) -> Tile((tx + (ty%2))//2, ty)
    # The distance between Tile(x1, y1) and Tile(x2, y2) in a grid is |x1 - x2| + |y1 - y2|.
    # For Sample 1: x1 = (5 + 0)//2 = 2, y1 = 0; x2 = (2 + 1)//2 = 1, y2 = 5.
    # Distance = |2 - 1| + |0 - 5| = 1 + 5 = 6.
    # But the sample says 5. Why?
    # Because he can move to (4,0) first, then (4,1).
    # (4,0) is in Tile((4+0)//2, 0) = (2, 0).
    # (4,1) is in Tile((4+1)//2, 1) = (2, 1).
    # The transition from (4,0) to (4,1) is a vertical move.
    # The tile at (4,0) is Tile(2,0). The tile at (4,1) is Tile(2,1).
    # This is a distance of |2-2| + |0-1| = 1.
    # Then from (4,1) to (3,1) is Tile(2,1) to Tile(2,1). Distance 0.
    # Then from (3,1) to (3,4) is Tile(2,1) to Tile(1,4). Distance |2-1| + |1-4| = 1 + 3 = 4.
    # Total = 1 + 4 = 5.
    
    # The key is that he can choose to move to either x=sx or x=sx-1 (if sx+sy is odd) 
    # to change his starting tile-x.
    # Actually, for any (sx, sy), he is in tile ((sx + (sy%2))//2, sy).
    # But he can move to (sx-1, sy) for free if (sx + (sy%2))//2 == ((sx-1) + (sy%2))//2.
    # The possible tile-x coordinates he can reach at height sy for free are:
    # If (sx + (sy%2)) is even, he can reach x_tile = (sx + (sy%2))//2 and (sx-1 + (sy%2))//2.
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # So for a fixed j, the tiles are {0,1}, {2,3}, {4,5}... if j is even.
    # And {1,2}, {3,4}, {5,6}... if j is odd.
    # This means i and i+1 are in the same tile if (i+j) is even.
    # This is exactly what (i + (j%2)) // 2 does.
    # For a fixed j, the cells i and i+1 are in the same tile if (i + (j%2)) // 2 == (i+1 + (j%2)) // 2.
    # This happens