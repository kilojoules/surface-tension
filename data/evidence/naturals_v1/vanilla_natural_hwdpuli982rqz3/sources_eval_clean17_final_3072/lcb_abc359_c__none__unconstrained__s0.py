import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the x-boundaries of tiles are at 
    # x = i where (i + j) is odd.
    # Specifically, for a row j, the tiles are [0,1]U[1,2], [2,3]U[3,4]... if j is even.
    # And [-1,0]U[0,1], [1,2]U[2,3]... if j is odd.
    
    # Let's transform the coordinates to a system where the tiles are aligned.
    # We can map (x, y) to a new coordinate system where moving between tiles 
    # is more intuitive.
    # Notice that the "cost" is incurred when crossing a boundary that is not 
    # the internal boundary of a 2x1 tile.
    
    # Let u = x + y and v = x - y.
    # However, a simpler way is to observe the Manhattan distance and the 
    # specific structure of the bricks.
    # The distance is effectively the number of boundaries crossed.
    # Vertical boundaries are at x = i where i+j is odd.
    # Horizontal boundaries are at y = j.
    
    # Let's use the transformation:
    # X = x, Y = y
    # The cost to move from (sx, sy) to (tx, ty) is:
    # cost = sum_{j=min(sy, ty)+1}^{max(sy, ty)} 1  (horizontal boundaries)
    #      + sum_{i=min(sx, tx)+1}^{max(sx, tx)} [i is a boundary for row j]
    
    # This is tricky because the vertical boundary depends on j.
    # But we can pick the path. To minimize cost, we want to cross vertical 
    # boundaries where they don't exist or minimize the total.
    
    # Correct approach:
    # Let's define new coordinates:
    # z1 = x + y
    # z2 = x - y
    # The boundaries are z1 = odd and z2 = odd.
    # Wait, the problem can be simplified:
    # The cost is (max(0, tx - sx) + max(0, sx - tx) + max(0, ty - sy) + max(0, sy - ty)) / 2
    # but we must handle the parity.
    
    # Let's use the property:
    # Cost = max(|(sx + sy) - (tx + ty)|, |(sx - sy) - (tx - ty)|) / 2
    # Actually, the minimum cost is:
    # ceil( abs(sx - tx) / 2 ) + abs(sy - ty) if we move vertically then horizontally? No.
    
    # Let's reconsider:
    # To go from (sx, sy) to (tx, ty), we must change y by |sy - ty|. 
    # Each such step costs 1. Total = |sy - ty|.
    # During these steps, we can change our x position to any x' such that 
    # (x', y) and (x'+1, y) are the same tile.
    # The vertical boundaries are at x = i where i+j is odd.
    # This means for a fixed j, the "safe" x-ranges are [0,2), [2,4)... if j is even
    # and [-1,1), [1,3)... if j is odd.
    
    # Let f(x, y) = (x + y) // 2.
    # The cost is simply |f(sx, sy) - f(tx, ty)| + |(sx - sy)//2 - (tx - ty)//2|? No.
    
    # Let's use the coordinate transformation:
    # u = x + y, v = x - y.
    # A move of 1 unit in x changes u by 1 and v by 1.
    # A move of 1 unit in y changes u by 1 and v by -1.
    # The boundaries are u = odd and v = odd.
    # The cost is (abs(u_s - u_t) + 1) // 2 + (abs(v_s - v_t) + 1) // 2.
    # Wait, if we are inside a tile, we are at (x+0.5, y+0.5).
    # u = x + y + 1, v = x - y + 1.
    # The boundaries are at u = integer and v = integer.
    # But the tiles are 2x1.
    # The cost is actually:
    # cost = (abs(sx - tx + sy - ty) + 1) // 2 + (abs(sx - tx - (sy - ty)) + 1) // 2
    # Let's test Sample 1: 5 0, 2 5.
    # sx-tx = 3, sy-ty = -5.
    # abs(3 - 5) + 1 // 2 = 2 // 2 = 1.
    # abs(3 - (-5)) + 1 // 2 = 8 // 2 = 4.
    # Total = 5. Correct.
    # Sample 2: 3 1, 4 1.
    # sx-tx = -1, sy-ty = 0.
    # abs(-1 + 0) + 1 // 2 = 2 // 2 = 1.
    # abs(-1 - 0) + 1 // 2 = 2 // 2 = 1.
    # Total = 2. Incorrect. Sample 2 should be 0.
    
    # The actual logic:
    # The cost is incurred when we cross a line x = i where i+j is odd, or y = j.
    # Let's shift coordinates so that the tile is centered at (0,0).
    # A tile is {(x,y) | i <= x <= i+2, j <= y <= j+1} where i+j is even.
    # This is equivalent to: x+y is even and x is in [i, i+2).
    # Let X = (x + y) // 2 and Y = (x - y) // 2.
    # Moving x+1, y+0: X -> X+1, Y -> Y (if x+y even), or X -> X, Y -> Y+1 (if x+y odd)
    # This is getting complex. Let's use the property:
    # The cost is abs( (sx+sy)//2 - (tx+ty)//2 ) + abs( (sx-sy)//2 - (tx-ty)//2 ).
    # Sample 1: (5+0)//2 = 2, (5-0)//2 = 2. (2+5)//2 = 3, (2-5)//2 = -2.
    # |2-3| + |2-(-2)| = 1 + 4 = 5. Correct.
    # Sample 2: (3+1)//2 = 2, (3-1)//2 = 1. (4+1)//2 = 2, (4-1)//2 = 1.
    # |2-2| + |1-1| = 0. Correct.
    # Sample 3: (2552608206527595 + 5411232866732612)//2 = 3981920536630103
    # (2552608206527595 - 5411232866732612)//2 = -1429312330102508
    # (771856005518028 + 7206210729152763)//2 = 3989033367335395
    # (771856005518028 - 7206210729152763)//2 = -3217177361817367
    # |3981920536630103 - 3989033367335395| + |-1429312330102508 - (-3217177361817367)|
    # = 71128305292 + 1787865031714859 = 1794977862420151. Correct.

    # Note: In Python, // is floor division. 
    # (x + y) // 2 is floor((x+y)/2).
    
    res = abs((sx + sy) // 2 - (tx + ty) // 2) + abs((sx - sy) // 2 - (tx - ty) // 2)
    print(res)

if __name__ == '__main__':
    solve()