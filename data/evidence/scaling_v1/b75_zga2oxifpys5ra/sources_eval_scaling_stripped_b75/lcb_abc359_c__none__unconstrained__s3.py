import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3)... if j is even
    # and (-1,0), (1,2)... if j is odd.
    # Essentially, a tile is defined by the pair (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a coordinate system where each unit 
    # represents a tile boundary.
    # In the y-direction, every move to a new j is a new tile.
    # In the x-direction, the "tile index" is (i + (j % 2)) // 2.
    
    # The cost to move between two tiles (x1, y1) and (x2, y2) in a grid 
    # where you can move any distance in one direction for the cost of 
    # the number of tiles entered:
    # This is equivalent to the L1 distance in the transformed space, 
    # but since we can move any n units, the cost is simply the 
    # difference in coordinates, provided we optimize the path.
    
    # Specifically, the cost is max(|x1 - x2|, |y1 - y2|) if we could move diagonally.
    # But we move axially. The cost is actually:
    # cost = abs(y1 - y2) + max(0, abs(x1 - x2) - abs(y1 - y2)) 
    # if the parity of the tiles allows "free" x-moves during y-moves.
    
    # Let's use the transformed coordinates:
    # X = (i + (j % 2)) // 2
    # Y = j
    
    # The distance between (X1, Y1) and (X2, Y2) in this specific tiling 
    # is known to be:
    # dist = abs(Y1 - Y2) + max(0, abs(X1 - X2) - (abs(Y1 - Y2) + 1) // 2) 
    # Wait, the simpler derivation for this specific problem is:
    # The cost is abs(Y1 - Y2) + max(0, abs(X1 - X2) - (abs(Y1 - Y2) // 2 + (1 if parity matches else 0)))
    
    # Let's use the coordinate transformation:
    # A point (i, j) belongs to tile ((i + (j % 2)) // 2, j)
    x1, y1 = (sx + (sy % 2)) // 2, sy
    x2, y2 = (tx + (ty % 2)) // 2, ty
    
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    
    # The minimum cost to move between these tiles is:
    # Each vertical step covers 1 unit of Y and can potentially 
    # cover 1 unit of X for 'free' every two vertical steps 
    # (because the tile boundaries shift).
    # The formula for this specific tiling distance is:
    # cost = dy + max(0, dx - (dy // 2 + (1 if (sx+sy)%2 != (tx+ty)%2 and dy%2==0 else 0)))
    # Actually, a more robust version:
    # The cost is dy + max(0, dx - (dy + 1) // 2) if we start/end on 
    # specific parities.
    
    # Correct logic for this problem:
    # The distance is dy + max(0, dx - (dy // 2 + (1 if (dy % 2 == 0 and 
    # ((sx + sy) % 2 != (tx + ty) % 2) else 0)))
    # Let's refine:
    # If we move dy vertically, we can cover dy // 2 horizontal tiles 
    # without extra cost, plus one more if the parity of the 
    # starting and ending tiles allows it.
    
    # The simplest correct formula for this problem is:
    # result = dy + max(0, dx - (dy + 1) // 2) 
    # But we must account for the parity of (sx+sy) and (tx+ty).
    # If (sx+sy)%2 == (tx+ty)%2, we can cover dy // 2 horizontal tiles.
    # If (sx+sy)%2 != (tx+ty)%2, we can cover (dy+1) // 2 horizontal tiles.
    
    # Let's use the property:
    # Cost = dy + max(0, dx - (dy // 2 + (1 if (sy % 2 != ty % 2) or ((sx+sy)%2 != (tx+ty)%2) else 0)))
    # Actually, the most reliable formula for this problem is:
    # ans = dy + max(0, dx - (dy // 2 + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0)))
    # Wait, let's test Sample 1: 5 0 -> 2 5. 
    # x1=(5+0)//2=2, y1=0; x2=(2+1)//2=1, y2=5.
    # dx=1, dy=5. ans = 5 + max(0, 1 - (5//2 + (1 if 1!=1 else 0))) = 5 + max(0, 1-2) = 5. Correct.
    # Sample 2: 3 1 -> 4 1.
    # x1=(3+1)//2=2, y1=1; x2=(4+1)//2=2, y2=1.
    # dx=0, dy=0. ans = 0 + max(0, 0-0) = 0. Correct.
    
    # Final formula:
    ans = dy + max(0, dx - (dy // 2 + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0)))
    print(ans)

if __name__ == "__main__":
    solve()