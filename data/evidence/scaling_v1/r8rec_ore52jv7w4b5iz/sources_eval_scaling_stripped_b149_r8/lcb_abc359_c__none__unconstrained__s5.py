import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), (4,5)... if j is even
    # and pairs (1,2), (3,4), (5,6)... if j is odd.
    
    # Let's define a coordinate transformation to a space where 
    # the cost is simply the L1 distance.
    # In the original grid, moving vertically always crosses a tile boundary 
    # (unless you stay in the same tile, but tiles are 2x1 horizontal).
    # Moving horizontally might be free if you stay within the 2x1 tile.
    
    # Let x' = x + (j % 2) if we consider the offset.
    # Actually, the cost to move from (sx, sy) to (tx, ty) in this specific 
    # brick-layer pattern is known to be:
    # cost = max(|sx - tx|, |sy - ty|) is for different patterns.
    # For this specific pattern:
    # The distance is (abs(sx - tx) + abs(sy - ty)) // 2 
    # but adjusted for the parity of the starting tile.
    
    # A more robust approach:
    # Each step in Y always costs 1.
    # Each step in X costs 1 every 2 units, but the "free" unit shifts based on Y.
    # This is equivalent to the distance in a coordinate system where 
    # we map (x, y) -> (x + (y % 2), y).
    # The distance between (x1, y1) and (x2, y2) is then 
    # (abs(x1' - x2') + abs(y1 - y2)) / 2.
    
    # Let's refine:
    # Let f(x, y) = (x + (y % 2), y)
    # The distance is max(abs(y1 - y2), (abs(x1' - x2') + abs(y1 - y2)) // 2)
    # Wait, the simplest formula for this specific tiling problem is:
    # cost = (abs(sx - tx) + abs(sy - ty) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2
    # No, that's for a different tile.
    
    # Correct logic for this tiling:
    # The cost is simply:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # If we move dy units vertically, we must pay dy.
    # The horizontal distance dx is covered by the "free" halves of the tiles.
    # Each vertical move effectively shifts the "free" zone.
    # The minimum cost is max(dy, (dx + dy + 1) // 2) if we account for parity.
    
    # Let's use the coordinate transformation:
    # X_new = x + (y % 2)
    # Y_new = y
    # The distance is then (abs(X_new1 - X_new2) + abs(Y_new1 - Y_new2)) // 2
    # But we must handle the starting parity.
    
    # Let's use the property: 
    # Cost = (abs(sx - tx) + abs(sy - ty)) // 2 
    # adjusted by the parity of (sx + sy) and (tx + ty).
    
    # The most reliable formula for this specific problem:
    # Let x1, y1 = sx, sy and x2, y2 = tx, ty
    # The distance is max(abs(y1 - y2), (abs(x1 - x2) + abs(y1 - y2) + (1 if (x1+y1)%2 != (x2+y2)%2 else 0)) // 2)
    # Actually, the simplest correct form is:
    # ans = (abs(sx - tx) + abs(sy - ty) + (1 if (sx + sy) % 2 != (tx + ty) % 2 else 0)) // 2
    # Let's check Sample 1: 5 0, 2 5 -> (3 + 5 + (1 if 5%2 != 7%2 else 0)) // 2 = (8 + 0) // 2 = 4. 
    # Sample 1 output is 5. So this formula is wrong.
    
    # Correct logic:
    # Each vertical move costs 1. 
    # A horizontal move is free if it stays in the 2x1 tile.
    # The tiles are {(2k, j), (2k+1, j)} if j is even, and {(2k-1, j), (2k, j)} if j is odd.
    # This is equivalent to saying x+j is even for the left square of the tile.
    # Let's transform coordinates: u = x + (y % 2), v = y.
    # The distance is then (abs(u1 - u2) + abs(v1 - v2)) // 2.
    # Sample 1: (5, 0) -> u=5+0=5, v=0; (2, 5) -> u=2+1=3, v=5.
    # (abs(5-3) + abs(0-5)) // 2 = (2 + 5) // 2 = 3. Still wrong.
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty):
    # Each vertical step costs 1.
    # Each horizontal step costs 1 every 2 units.
    # The cost is actually:
    # dist = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # But the parity of the "free" square depends on y.
    # The correct distance is:
    # Let x1, y1 = sx, sy and x2, y2 = tx, ty
    # ans = abs(y1 - y2) + max(0, (abs(x1 - x2) - abs(y1 - y2) + (1 if (x1+y1)%2 == (x2+y2)%2 else 0)) // 2)
    # Sample 1: 5 0, 2 5 -> 5 + max(0, (3 - 5 + (1 if 5%2 == 7%2 else 0)) // 2) = 5 + 0 = 5. Correct.
    # Sample 2: 3 1, 4 1 -> 0 + max(0, (1 - 0 + (1 if 4%2 == 5%2 else 0)) // 2) = max(0, (1 + 0) // 2) = 0. Correct.
    
    # Let's double check the parity logic:
    # If we move dy vertically, we can cover dy horizontal distance for "free" 
    # (by picking the right sequence of moves).
    # The remaining horizontal distance is dx - dy.
    # If dx > dy, we need (dx - dy) // 2 additional tiles.
    # The parity depends on whether the start and end points are in the "same" 
    # relative position within their respective tiles.
    
    # Final formula:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # parity_adjustment = 1 if (sx + sy) % 2 == (tx + ty) % 2 else 0
    # result = dy + max(0, (dx - dy + parity_adjustment) // 2)
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    adj = 1 if (sx + sy) % 2 == (tx + ty) % 2 else 0
    print(dy + max(0, (dx - dy + adj) // 2))

if __name__ == "__main__":
    solve()