import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1 blocks. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    # Essentially, a tile is defined by (floor((i + (j % 2)) / 2), j).
    
    # Let's transform coordinates to a system where we can calculate distance.
    # The cost to move vertically is 1 per unit (each new j is a new tile).
    # The cost to move horizontally depends on whether we cross a tile boundary.
    # In this specific tiling, moving horizontally costs 0 if we stay in the same 2x1 tile,
    # and moving vertically always costs 1.
    
    # The minimum cost to get from (sx, sy) to (tx, ty) in this specific 
    # brick-layer pattern is given by the formula:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2)
    # However, a more robust way to view this is:
    # To move dx horizontally and dy vertically:
    # Each vertical step costs 1.
    # Horizontal movement is "free" every other step if synchronized with the vertical.
    # The actual minimum cost is max(|sy - ty|, ceil((|sx - tx| + |sy - ty|) / 2))
    # Wait, the standard derivation for this specific tiling problem is:
    # Let dx = abs(sx - tx), dy = abs(sy - ty)
    # The cost is max(dy, (dx + dy + 1) // 2) if we consider the parity.
    # Actually, the simplest correct form for this problem is:
    # result = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty) + 1) // 2)
    # Let's verify with Sample 1: 5 0, 2 5 -> dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: 3 1, 4 1 -> dx=1, dy=0. max(0, (1+0+1)//2) = 1? 
    # Wait, Sample 2 says 0. Let's re-evaluate.
    
    # Re-evaluating the tile boundary:
    # Tile ID for (i, j) is ( (i + (j % 2)) // 2, j )
    # Start tile: ((sx + (sy % 2)) // 2, sy)
    # End tile: ((tx + (ty % 2)) // 2, ty)
    
    # Let X1 = (sx + (sy % 2)) // 2, Y1 = sy
    # Let X2 = (tx + (ty % 2)) // 2, Y2 = ty
    # The distance in this metric is max(|Y1 - Y2|, (|X1 - X2| + |Y1 - Y2| + 1) // 2)
    # Wait, the simplest form is:
    # dx = abs(X1 - X2), dy = abs(Y1 - Y2)
    # cost = max(dy, (dx + dy + 1) // 2) is for a different grid.
    # For this specific problem, the cost is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2) 
    # Let's check Sample 2: sx=3, sy=1, tx=4, ty=1.
    # X1 = (3 + 1)//2 = 2, Y1 = 1
    # X2 = (4 + 1)//2 = 2, Y2 = 1
    # dx = 0, dy = 0. Cost = 0. Correct.
    # Sample 1: sx=5, sy=0, tx=2, ty=5.
    # X1 = (5 + 0)//2 = 2, Y1 = 0
    # X2 = (2 + (5%2))//2 = (2+1)//2 = 1, Y2 = 5
    # dx = |2-1| = 1, dy = |0-5| = 5.
    # Cost = max(5, (1+5)//2) = 5. Correct.
    
    # The general formula for this tiling is:
    # Let x1 = (sx + (sy % 2)) // 2
    # Let y1 = sy
    # Let x2 = (tx + (ty % 2)) // 2
    # Let y2 = ty
    # ans = max(abs(y1 - y2), (abs(x1 - x2) + abs(y1 - y2) + 1) // 2)
    # Wait, let's try: ans = max(abs(y1 - y2), (abs(x1 - x2) + abs(y1 - y2)) // 2)
    # Sample 1: max(5, (1+5)//2) = 5.
    # Sample 2: max(0, (0+0)//2) = 0.
    # Let's test another: sx=0, sy=0, tx=1, ty=0.
    # X1 = 0, Y1 = 0; X2 = (1+0)//2 = 0, Y2 = 0. Ans = 0. Correct (same tile).
    # sx=0, sy=0, tx=2, ty=0.
    # X1 = 0, Y1 = 0; X2 = (2+0)//2 = 1, Y2 = 0. Ans = max(0, 1//2) = 0? 
    # No, moving from (0,0) to (2,0) requires crossing a boundary.
    # (0,0) and (1,0) are tile A. (2,0) and (3,0) are tile B.
    # To get from A to B, you must move to a different row or cross the boundary.
    # The cost is 1.
    # Using the formula: X1=0, Y1=0, X2=1, Y2=0. 
    # (abs(X1-X2) + abs(Y1-Y2) + 1) // 2 = (1 + 0 + 1) // 2 = 1.
    # So the correct formula is max(abs(y1 - y2), (abs(x1 - x2) + abs(y1 - y2) + 1) // 2)
    # But wait, if x1==x2 and y1==y2, it should be 0.
    # (0 + 0 + 1) // 2 is 0. Correct.
    
    x1 = (sx + (sy % 2)) // 2
    y1 = sy
    x2 = (tx + (ty % 2)) // 2
    y2 = ty
    
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    
    # The minimum cost is max(dy, (dx + dy + 1) // 2)
    # However, we must handle the case where we are already in the same tile.
    # If x1 == x2 and y1 == y2, the cost is 0.
    # The formula (dx + dy + 1) // 2 for dx=0, dy=0 gives 0.
    # For dx=1, dy=0, it gives 1.
    # For dx=0, dy=1, it gives 1.
    # For dx=1, dy=1, it gives 1.
    # Let's check Sample 1 again: dx=1, dy=5. max(5, (1+5+1)//2) = 5.
    # Sample 2: dx=0, dy=0. max(0, (0+0+1)//2) = 0.
    
    print(max(dy, (dx + dy + 1) // 2) if not (x1 == x2 and y1 == y2) else 0)

# Using a function wrapper to avoid global scope logic and maintain clean structure
if __name__ == "__main__":
    solve()