import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed j, tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    
    # Let's normalize the coordinates to a grid of 2x1 blocks.
    # A tile covers two squares: (i, j) and (i+1, j) if i+j is even.
    # This is equivalent to saying a tile is defined by (floor((i + (j % 2)) / 2), j).
    
    # Let X' = (x + (y % 2)) // 2
    # Let Y' = y
    # Moving in Y direction always crosses a tile boundary (cost 1).
    # Moving in X direction might stay in the same tile or cross one.
    
    # However, the cost is simply the distance in a transformed coordinate system.
    # The distance is max(|X1' - X2'|, |Y1' - Y2'|) is not quite right because
    # Y movement is strictly vertical.
    
    # Correct logic:
    # To move from (sx, sy) to (tx, ty):
    # The cost is the number of tile boundaries crossed.
    # Vertical movement: each step in y crosses a boundary. Cost = |sy - ty|.
    # Horizontal movement: 
    # In a row j, tiles are [0,1], [2,3]... if j is even.
    # In a row j, tiles are [-1,0], [1,2]... if j is odd.
    # Let's map x to a tile index: tile_x = (x + (j % 2)) // 2.
    # The distance is then related to the change in tile_x and the change in j.
    
    # The minimum cost to get from (sx, sy) to (tx, ty) is:
    # cost = max(|sy - ty|, (abs(sx - tx) + (1 if parity changes and we are at the edge else 0)) // 2)
    # Actually, the simplest form for this specific tiling is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Wait, the standard formula for this specific problem (often found in competitive programming) is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The cost is max(dy, (dx + 1) // 2) if we account for the shift.
    # More accurately:
    # If we move from (sx, sy) to (tx, ty), the cost is:
    # max(abs(sy - ty), (abs(sx - tx) + (1 if (sx % 2 != (tx % 2 if sy % 2 == ty % 2 else (tx+1) % 2)) else 0)) // 2)
    
    # Let's use the coordinate transformation:
    # A square (x, y) belongs to tile ( (x + (y % 2)) // 2, y )
    # Let X1 = (sx + (sy % 2)) // 2, Y1 = sy
    # Let X2 = (tx + (ty % 2)) // 2, Y2 = ty
    # The distance is max(abs(X1 - X2), abs(Y1 - Y2))
    # But wait, moving in Y changes the X coordinate of the tile.
    # The actual minimum cost is max(abs(sy - ty), (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Let's re-verify: if sx=5, sy=0, tx=2, ty=5.
    # X1 = (5+0)//2 = 2, Y1 = 0
    # X2 = (2+1)//2 = 1, Y2 = 5
    # max(|2-1|, |0-5|) = 5. Correct.
    # Sample 2: 3 1, 4 1.
    # X1 = (3+1)//2 = 2, Y1 = 1
    # X2 = (4+1)//2 = 2, Y2 = 1
    # max(0, 0) = 0. Correct.
    
    # The logic is:
    # The distance is max(abs(sy - ty), (abs(sx - tx) + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0)) // 2)
    # Wait, the parity logic is simpler:
    # Let's use the transformation:
    # x_tile = (x + (y % 2)) // 2
    # y_tile = y
    # The distance is max(abs(x_tile1 - x_tile2), abs(y_tile1 - y_tile2))
    # But since moving in y changes the parity, it can shift the x_tile.
    # The most robust formula for this problem is:
    # ans = max(abs(sy - ty), (abs(sx - tx) + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0)) // 2)
    # Let's check Sample 1 again: sx=5, sy=0, tx=2, ty=5.
    # abs(0-5) = 5. (abs(5-2) + (1 if 5%2 != 7%2 else 0)) // 2 = (3 + 0) // 2 = 1.
    # max(5, 1) = 5.
    
    # Let's check a case where X dominates: sx=0, sy=0, tx=10, ty=0.
    # X1 = 0, Y1 = 0. X2 = 5, Y2 = 0.
    # max(0, (10 + 0)//2) = 5.
    # Manual check: (0,0)-(1,0) is tile 1. (2,0)-(3,0) is tile 2... (10,0)-(11,0) is tile 6.
    # To get from square 0 to square 10, you cross tiles 1, 2, 3, 4, 5, 6.
    # Starting in tile 1, you enter 5 more tiles. Cost = 5.
    
    # Final formula:
    # x1 = (sx + (sy % 2)) // 2
    # y1 = sy
    # x2 = (tx + (ty % 2)) // 2
    # y2 = ty
    # result = max(abs(x1 - x2), abs(y1 - y2))
    
    # Wait, if sy and ty have different parity, the x_tile mapping shifts.
    # The correct distance in this metric is indeed max(abs(x1-x2), abs(y1-y2))
    # because a move in Y by 1 unit costs 1 and can potentially change the x_tile 
    # by 0 or 1, and a move in X by 2 units costs 1.
    
    # Let's double check Sample 1 with this:
    # sx=5, sy=0 -> x1 = (5+0)//2 = 2, y1 = 0
    # tx=2, ty=5 -> x2 = (2+1)//2 = 1, y2 = 5
    # max(|2-1|, |0-5|) = 5.
    
    # Sample 2:
    # sx=3, sy=1 -> x1 = (3+1)//2 = 2, y1 = 1
    # tx=4, ty=1 -> x2 = (4+1)//2 = 2, y2 = 1
    # max(0, 0) = 0.
    
    # This logic holds.
    
    print(max(abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2), abs(sy - ty)))

if __name__ == "__main__":
    solve()