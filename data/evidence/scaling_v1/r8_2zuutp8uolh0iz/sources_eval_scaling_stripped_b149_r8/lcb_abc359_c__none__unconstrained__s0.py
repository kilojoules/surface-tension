import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling pattern:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even, tiles are [0,1][2,3]... (horizontal pairs starting at even i)
    # If j is odd, tiles are [1,2][3,4]... (horizontal pairs starting at odd i)
    
    # Let's define a coordinate transformation to a space where 
    # the cost is simply the L1 distance.
    # In the original grid, moving vertically always crosses a tile boundary 
    # unless you are already in the target tile.
    # Moving horizontally might be free if you stay within the 2x1 tile.
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # cost = abs(sy - ty) + (cost to move from sx to tx given the parity of sy and ty)
    
    # Let dx = tx - sx and dy = ty - sy.
    # The total vertical distance is abs(dy).
    # The horizontal cost depends on the relative positions.
    # If we move vertically, we can 'shift' our horizontal alignment.
    
    # A more robust approach:
    # Let f(x, y) = (x + (y % 2)) // 2
    # The distance is essentially max(abs(sy - ty), abs(f(sx, sy) - f(tx, ty)))
    # Wait, the actual distance formula for this specific tiling is:
    # cost = abs(sy - ty) + max(0, abs( (sx + (sy%2))//2 - (tx + (ty%2))//2 ) - abs(sy - ty)//2 )
    # Actually, the simplest correct formula for this problem is:
    # Let X = (sx + (sy % 2)) // 2 and Y = (tx + (ty % 2)) // 2
    # The distance is max(abs(sy - ty), abs(X - Y))
    # But we must account for the parity of the vertical distance.
    
    # Correct logic:
    # Each vertical step costs 1.
    # Each horizontal step costs 1 every 2 units, but the "offset" changes every row.
    # The distance is:
    # dist = abs(sy - ty)
    # horizontal_dist = abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    # However, vertical moves can cover some horizontal distance.
    # Specifically, moving vertically for 'dist' units allows you to 
    # change your horizontal tile index by at most ceil(dist/2).
    
    # The general formula for this specific tiling problem is:
    # ans = max(abs(sy - ty), abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2))
    # Let's verify with Sample 1: 5 0 -> 2 5
    # sy=0, ty=5. abs(sy-ty) = 5.
    # X = (5 + 0)//2 = 2. Y = (2 + (5%2))//2 = (2+1)//2 = 1.
    # abs(X-Y) = 1.
    # max(5, 1) = 5. Correct.
    
    # Sample 2: 3 1 -> 4 1
    # sy=1, ty=1. abs(sy-ty) = 0.
    # X = (3 + 1)//2 = 2. Y = (4 + 1)//2 = 2.
    # max(0, 0) = 0. Correct.
    
    # Sample 3: 2552608206527595 5411232866732612 -> 771856005518028 7206210729152763
    # sy=5411232866732612, ty=7206210729152763
    # abs(sy-ty) = 1794977862420151
    # X = (2552608206527595 + 0)//2 = 1276304103263797
    # Y = (771856005518028 + 1)//2 = 385928002759014
    # abs(X-Y) = 890376100504783
    # max(1794977862420151, 890376100504783) = 1794977862420151. Correct.

    # Using reduce to avoid loops/recursion as per constraints (though not needed for this formula)
    result = reduce(lambda acc, val: max(acc, val), [
        abs(sy - ty),
        abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2)
    ], 0)
    
    print(result)

if __name__ == "__main__":
    solve()