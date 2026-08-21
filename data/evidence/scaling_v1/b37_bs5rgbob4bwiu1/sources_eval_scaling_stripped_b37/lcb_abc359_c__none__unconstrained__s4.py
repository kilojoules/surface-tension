import sys

def solve():
    # Read input and parse integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1 rectangles. 
    # A square (i, j) and (i+1, j) are in the same tile if i+j is even.
    # This means for a fixed j:
    # If j is even, pairs are (0,1), (2,3), (4,5)...
    # If j is odd, pairs are (-1,0), (1,2), (3,4)...
    
    # We can represent the state by (x, y). 
    # A move in x costs 0 if we stay within the same 2x1 tile.
    # A move in y always costs 1 per unit distance because every 
    # vertical step enters a new tile.
    # However, the cost depends on the parity of the coordinates.
    
    # Let's use a coordinate transformation to simplify the grid.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # Vertical distance: abs(sy - ty)
    # Horizontal distance: depends on whether the start and end 
    # squares are in the same tile or different tiles.
    
    # A square (x, y) belongs to a tile identified by:
    # TileID = ( (x // 2) if (x + y) % 2 == 0 else (x - 1) // 2 , y )
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even, tiles are [0,1], [2,3], [4,5]...
    # If j is odd, tiles are [-1,0], [1,2], [3,4]...
    
    # Let f(x, y) be the index of the tile in row y.
    # If y % 2 == 0: tile_index = x // 2
    # If y % 2 == 1: tile_index = (x - 1) // 2 if x > 0 else -1
    # More generally: tile_index = (x + (y % 2)) // 2
    
    # The distance is the L1 distance in the transformed space:
    # Let X = (x + (y % 2)) // 2
    # Let Y = y
    # The cost is abs(X1 - X2) + abs(Y1 - Y2)
    # But we must check if the parity of y changes the X coordinate.
    # Actually, the cost is simply:
    # cost = abs(sy - ty) + max(0, abs(X1 - X2) - (1 if (sx+sy)%2 == (tx+ty)%2 else 0))
    # That's not quite right. Let's use the property:
    # The distance is abs(sy - ty) + distance_x
    # where distance_x is the number of horizontal tile boundaries crossed.
    
    # Correct logic:
    # Let g(x, y) = (x + (y % 2)) // 2
    # The minimum cost is abs(g(sx, sy) - g(tx, ty)) + abs(sy - ty)
    # However, if we move vertically, we might change the 'column' g(x, y)
    # without paying a horizontal toll.
    
    # Let's refine:
    # The cost is abs(sy - ty) + max(0, abs(g(sx, sy) - g(tx, ty)) - 1)
    # Wait, if we are at (sx, sy) and (tx, ty), and we move to the same 
    # 'column' g, the cost is just the vertical distance.
    # If we are in different columns, we must pay for the horizontal shift.
    # The only way to reduce the cost is if the vertical movement 
    # allows us to 'shift' our column index for free.
    
    # Let's use the parity logic:
    # The cost is abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # But we can reduce this by 1 if (sx + sy) % 2 == 0 and (tx + ty) % 2 == 0
    # AND we move in a way that we utilize the tile width.
    # Actually, the simplest correct formula for this specific problem is:
    # ans = abs(sy - ty) + max(0, abs(g(sx, sy) - g(tx, ty)) - (1 if (sx+sy)%2 == 0 and (tx+ty)%2 == 0 else 0))
    # No, that's for a different version.
    
    # The correct logic for this grid:
    # The distance is abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # If (sx + sy) % 2 == 0 and (tx + ty) % 2 == 0, we can potentially save 1.
    # Let's test Sample 1: 5 0 -> 2 5
    # g(5, 0) = (5 + 0) // 2 = 2
    # g(2, 5) = (2 + 1) // 2 = 1
    # cost = abs(0 - 5) + abs(2 - 1) = 5 + 1 = 6. 
    # Sample 1 output is 5. So we save 1.
    # (5+0)%2 = 1, (2+5)%2 = 1. Both are odd.
    
    # If (sx + sy) % 2 == (tx + ty) % 2, we can save 1 toll.
    # Let's check Sample 2: 3 1 -> 4 1
    # g(3, 1) = (3 + 1) // 2 = 2
    # g(4, 1) = (4 + 1) // 2 = 2
    # cost = abs(1 - 1) + abs(2 - 2) = 0. Correct.
    
    # Final logic:
    # res = abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # if (sx + sy) % 2 == (tx + ty) % 2: res -= 1
    # But res cannot be negative.
    # Wait, if sx=0, sy=0 and tx=0, ty=0, res = 0 + 0 - 1 = -1.
    # Also, if sx=0, sy=0 and tx=1, ty=0, g(0,0)=0, g(1,0)=0, res = 0 + 0 - 1 = -1.
    # But the toll is 0.
    
    # Let's reconsider:
    # The cost is abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # We subtract 1 if (sx + sy) % 2 == (tx + ty) % 2, 
    # UNLESS we are already in the same tile.
    # Two points are in the same tile if sy == ty and g(sx, sy) == g(tx, ty).
    
    # Actually, the parity rule is: 
    # If (sx + sy) % 2 == (tx + ty) % 2, we can save 1 toll 
    # PROVIDED that we actually moved (distance > 0).
    # If we are in the same tile, the cost is 0.
    
    # Let'0s check Sample 1 again: (5,0) and (2,5).
    # g(5,0) = 2, g(2,5) = 1.
    # Dist = abs(0-5) + abs(2-1) = 6.
    # (5+0)%2 = 1, (2+5)%2 = 1. Parity same, so 6 - 1 = 5.
    
    # Sample 2: (3,1) and (4,1).
    # g(3,1) = 2, g(4,1) = 2.
    # Dist = abs(1-1) + abs(2-2) = 0.
    # (3+1)%2 = 0, (4+1)%2 = 1. Parity different.
    # Result = 0.
    
    # What if (0,0) and (1,0)?
    # g(0,0) = 0, g(1,0) = 0.
    # Dist = 0 + 0 = 0.
    # Parity: (0+0)%2 = 0, (1+0)%2 = 1. Different.
    # Result = 0.
    
    # What if (0,0) and (0,0)?
    # Dist = 0. Parity same. 0 - 1 = -1? No, should be 0.
    
    # Correct logic:
    # The cost is abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # If (sx + sy) % 2 == (tx + ty) % 2, we can reduce the cost by 1,
    # but the minimum possible cost is 0.
    # However, we only reduce if we actually crossed a boundary.
    # If we are in the same tile, the cost is 0.
    # If we are in different tiles, and parity is same, we save 1.
    
    # Let's refine:
    # If (sx, sy) and (tx, ty) are in the same tile, cost = 0.
    # Otherwise, cost = abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # And if (sx + sy) % 2 == (tx + ty) % 2, cost -= 1.
    
    # Check: (0,0) and (1,0) -> same tile?
    # g(0,0)=0, g(1,0)=0, sy=ty=0. Yes. Cost = 0.
    # (0,0) and (0,1) -> same tile?
    # g(0,0)=0, g(0,1)=0, sy=0, ty=1. No.
    # Dist = abs(0-1) + abs(0-0) = 1.
    # Parity: (0+0)%2=0, (0+1)%2=1. Different.
    # Cost = 1.
    
    # (0,0) and (1,1) -> same tile?
    # g(0,0)=0, g(1,1)=1, sy=0, ty=1. No.
    # Dist = abs(0-1) + abs(0-1) = 2.
    # Parity: (0+0)%2=0, (1+1)%2=0. Same.
    # Cost = 2 - 1 = 1.
    
    # Final implementation:
    g_s = (sx + (sy % 2)) // 2
    g_t = (tx + (ty % 2)) // 2
    
    if sy == ty and g_s == g_t:
        print(0)
    else:
        ans = abs(sy - ty) + abs(g_s - g_t)
        if (sx + sy) % 2 == (tx + ty) % 2:
            ans -= 1
        print(max(0, ans))

if __name__ == "__main__":
    solve()