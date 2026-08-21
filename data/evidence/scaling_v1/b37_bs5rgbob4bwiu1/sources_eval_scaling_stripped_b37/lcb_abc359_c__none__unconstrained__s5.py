```python
import sys

def solve():
    # Read input and map to integers
    # Sx, Sy: Starting coordinates
    # Tx, Ty: Target coordinates
    # The problem defines tiles based on the parity of i + j.
    # A tile covers A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j, the tiles are horizontal dominoes:
    # If j is even, tiles are {(0,1), (2,3), (4,5)...}
    # If j is odd, tiles are {(-1,0), (1,2), (3,4)...}
    
    # Let's normalize the coordinates to a grid where each cell is a tile.
    # A tile can be identified by (block_x, block_y).
    # For a given y, the x-boundary of tiles shifts.
    # If y is even, tiles are [0,1], [2,3]... so block_x = x // 2
    # If y is odd, tiles are [-1,0], [1,2]... so block_x = (x + 1) // 2
    # However, the parity of the block_x changes based on y.
    # To make it a consistent grid, we can transform (x, y) -> (X, Y)
    # where X = (x + (y % 2)) // 2 and Y = y.
    # In this transformed grid, moving 1 unit in Y always enters a new tile.
    # Moving 1 unit in X enters a new tile.
    # The distance is then the Manhattan distance in the (X, Y) space.
    # But wait, the cost is the number of tiles ENTERED.
    # The starting tile is free.
    # The cost is simply the Manhattan distance between the tile indices.
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])
    
    # Transform coordinates to tile indices (X, Y)
    # X = (x + (y % 2)) // 2
    # Y = y
    # The distance is |X1 - X2| + |Y1 - Y2|
    # However, we must account for the fact that moving in Y might 
    # land us in a tile that covers the current X.
    # Actually, the simplest way to view this is:
    # Every vertical move of 1 unit costs 1.
    # Every horizontal move of 1 unit in the 'tile-grid' costs 1.
    # The parity of y affects the x-offset.
    
    def get_tile_coords(x, y):
        # If y is even, tiles are [0,1], [2,3]...
        # If y is odd, tiles are [-1,0], [1,2]...
        # We can use (x + (y % 2)) // 2 to get the tile index
        return (x + (y % 2)) // 2, y

    s_tile = get_tile_coords(sx, sy)
    t_tile = get_tile_coords(tx, ty)
    
    # The distance is |Xs - Xt| + |Ys - Yt|
    # But there is a catch: moving vertically might change the X index.
    # If we move from (sx, sy) to (sx, ty), the X index might change.
    # The minimum cost is the minimum of:
    # 1. Move to (sx, ty) then to (tx, ty)
    # 2. Move to (tx, sy) then to (tx, ty)
    # Actually, the Manhattan distance in the transformed space 
    # |Xs - Xt| + |Ys - Yt| is almost correct, but we need to check
    # if the parity of y changes the X coordinate.
    
    # Let's use the property: 
    # Cost = abs(sy - ty) + abs(get_tile_x(sx, sy) - get_tile_x(tx, ty))
    # Wait, the parity of sy and ty matters.
    # If we move vertically, we might enter a tile that already covers our X.
    # The correct logic for this specific tiling is:
    # The distance is abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    # But we must consider that we can move to a y where the x-tile index is more favorable.
    # The parity of y only has two states.
    
    # Let f(x, y) = (x + (y % 2)) // 2
    # We want to minimize |y1 - y2| + |f(x1, y1) - f(x2, y2)|
    # Since we can move to y or y+1, we check both parities for the start and end.
    # Actually, the movement is: you are at (sx, sy). You can move to (sx, sy+1) 
    # and your tile index X changes from (sx + (sy%2))//2 to (sx + ((sy+1)%2))//2.
    
    # The most robust way:
    # The cost is abs(sy - ty) + abs(f(sx, sy) - f(tx, ty))
    # But we can potentially reduce the X-distance by 1 if we change the parity of y.
    # However, changing parity of y costs 1.
    # So we check the distance using the given sy and ty, 
    # and also check if moving to sy+1 or ty+1 helps.
    # Actually, the simplest mathematical expression for this problem is:
    # ans = abs(sy - ty) + abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    # Let's test this with Sample 1: 5 0 -> 2 5
    # sy=0, ty=5. abs(0-5) = 5.
    # f(5, 0) = (5+0)//2 = 2.
    # f(2, 5) = (2+1)//2 = 1.
    # abs(2-1) = 1. Total = 6. 
    # Sample 1 output is 5. Why?
    # Because we can move to a different y first.
    # If we move to y=1, f(5, 1) = (5+1)//2 = 3.
    # Then distance to f(2, 5)=1 is abs(3-1)=2. Total = 1 + 2 + 4 = 7.
    # If we move to y=0, then x=2, then y=5:
    # (5,0) tile X=2. (2,0) tile X=1. Cost = abs(2-1) = 1.
    # Then (2,0) to (2,5). Cost = abs(0-5) = 5. Total = 6.
    # Wait, the sample says 5. Let's re-read.
    # "Move left by 1. Pay 0." -> (5,0) to (4,0). Both are in tile X=2.
    # "Move up by 1. Pay 1." -> (4,1). Tile X=(4+1)//2 = 2.
    # "Move left by 1. Pay 0." -> (3,1). Tile X=(3+1)//2 = 2.
    # "Move up by 3. Pay 3." -> (3,4).
    # This means the cost is simply the Manhattan distance in the 
    # transformed coordinate system, but we must account for the 
    # fact that we can change our X-index by moving vertically.
    
    # Correct logic:
    # The cost is abs(sy - ty) + abs(f(sx, sy) - f(tx, ty))
    # But we can also reach the target by changing the parity of sy or ty.
    # The distance is min(
    #    abs(sy - ty) + abs(f(sx, sy) - f(tx, ty)),
    #    abs(sy - ty) + 1 + abs(f(sx, sy+1) - f(tx, ty)), # Not possible, moving y costs 1
    # )
    # Actually, the parity of y only affects X by 0.5.
    # The distance is abs(sy - ty) + abs((sx + (sy%2))//2 - (tx + (ty%2))//2)
    # Let's re-evaluate Sample 1: sx=5, sy=0, tx=2, ty=5
    # f(5,0) = 2, f(2,5) = 1. Dist = 5 + |2-1| = 6.
    # If we move to (5,1), cost 1. f(5,1) = 3. Dist to f(2,5)=1 is 2. Total 1+2+4=7.
    # If we move to (4,0), cost 0. f(4,0) = 2.
    # Wait, the sample says: (5,0) -> (4,0) [0] -> (4,1) [1] -> (3,1) [0] -> (3,4) [3] -> (2,4) [0] -> (2,5) [1].
    # Total = 0+1+0+3+0+1 = 5.
    # Let's trace tiles:
    # (5,0): X=(5+0)//2 = 2
    # (4,0): X=(4+0)//2 = 2
    # (4,1): X=(4+1)//2 = 2
    # (3,1): X=(3+1)//2 = 2
    # (3,4): X=(3+0)//2 = 1
    # (2,4): X=(2+0)//2 = 1
    # (2,5): X=(2+1)//2 = 1
    # The X indices were: 2 -> 2 -> 2 -> 2 -> 1 -> 1 -> 1.
    # The Y indices were: 0 -> 0 -> 1 -> 1 -> 4 -> 4 -> 5.
    # Total cost = (4-0) + (5-4) + (2-1) = 4 + 1 + 1 = 6? No.
    # The cost is the number of tiles ENTERED.
    # Start tile is free.
    # Move (5,0) to (4,0): same tile.
    # Move (4,0) to (4,1): new tile. (Cost 1)
    # Move (4,1) to (3,1): same tile.
    # Move (3,1) to (3,4): tiles at y=2,3,4. (Cost 3)
    # Move (3,4) to (2,4): same tile.
    # Move (2,4) to (2,5): new tile. (Cost 1)
    # Total = 1 + 3 + 1 = 5.
    
    # In this path, X indices were: 2, 2, 2, 2, 1, 1, 1.
    # Y indices were: 0, 0, 1, 1, 4, 4, 5.
    # The cost is simply abs(f(sx, sy) - f(tx, ty)) + abs(sy - ty)
    # BUT, we can choose to change our X index by moving to a different Y.
    # If we are at (x, y), our tile is X = (x + (y%2)) // 2.
    # If we move to (x, y+1), our new tile is X' = (x + ((y+1)%2)) // 2.
    # Note that X' is either X or X+1 (if y%2==0) or X-1 (if y%2==1).
    # This means we can change our X index by 1 by spending 1 Y-unit.
    # This is exactly like moving in a grid where you can move (X, Y) -> (X+/-1, Y) or (X, Y+/-1).
    # The only difference is that moving Y might automatically change X.
    # But we can always choose to move Y in a way that X stays the same or changes.
    # Actually, the rule is: you pay 1 every time you enter a NEW tile.
    # If you move from (X, Y) to (X, Y+1), you always enter a new tile.
    # If you move from (X, Y) to (X+1, Y), you always enter a new tile.
    # The only way