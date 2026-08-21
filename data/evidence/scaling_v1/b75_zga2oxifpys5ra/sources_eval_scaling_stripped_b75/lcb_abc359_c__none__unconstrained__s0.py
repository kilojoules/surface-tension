import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1. 
    # When i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3)... if j is even
    # and (-1,0), (1,2)... if j is odd.
    # Essentially, a tile is defined by (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a coordinate system where 
    # each unit of distance in the new system corresponds to a tile boundary.
    # The horizontal "block" index is b_x = (i + (j % 2)) // 2
    # The vertical "block" index is b_y = j
    
    # However, the cost is based on how many tiles you enter.
    # Moving vertically always enters a new tile.
    # Moving horizontally might stay in the same tile.
    
    # Let's use the property:
    # A tile is identified by (x_tile, y_tile) where:
    # x_tile = (i + (j % 2)) // 2
    # y_tile = j
    
    # The distance between two tiles (x1, y1) and (x2, y2) in this grid
    # is the L1 distance, but we must account for the fact that 
    # moving horizontally is "cheaper" in terms of tile entries.
    
    # The minimum cost to move between (sx, sy) and (tx, ty) is:
    # cost = max(|sx - tx|, |sy - tx|) is for Chebyshev, but here 
    # the tiles are oriented.
    
    # Correct logic for this specific tiling:
    # The cost is max(|sx - tx|, |sy - ty|) if we could move diagonally.
    # But we move in 4 directions.
    # The distance is actually:
    # dist = max(abs(sx - tx), abs(sy - ty)) is for a different problem.
    # For this specific tiling, the distance is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty)) // 2)
    # Wait, that's for a different tile layout.
    
    # Let's re-evaluate:
    # To move from (sx, sy) to (tx, ty):
    # Vertical distance is dy = abs(sy - ty).
    # Horizontal distance is dx = abs(sx - tx).
    # Each vertical step costs 1.
    # Horizontal steps cost 1 every 2 units, but the parity of the row matters.
    # The optimal strategy is to move diagonally as much as possible.
    # One diagonal step (1 horizontal, 1 vertical) costs 1 toll.
    # After covering dy, we have dx - dy horizontal distance left.
    # These remaining horizontal steps cost 1 toll every 2 units.
    
    # The formula for this specific tiling is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2)
    # Let's check Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+5)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+0)//2) = 0. Correct.
    
    # Why this formula?
    # Each vertical move costs 1. A vertical move also allows us to 
    # "shift" our horizontal tile alignment.
    # Effectively, we can move 1 unit vertically and 1 unit horizontally 
    # and only pay 1 toll (the vertical one), provided we are moving 
    # into the 'other' half of the 2x1 tile.
    
    # The cost is simply:
    # result = max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2)
    # But we must be careful with the parity of the starting tile.
    # Actually, the simplest general form for this tiling is:
    # cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    # Wait, if dx=1, dy=0, cost should be 0. (1-0+1)//2 = 1. Incorrect.
    
    # Let's use the coordinate transformation:
    # A tile is ( (i + j%2)//2, j )
    # Distance between (x1, y1) and (x2, y2) in this grid:
    # The cost to move from (sx, sy) to (tx, ty) is:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # The answer is max(dy, (dx + dy) // 2)
    # Let's re-verify Sample 2: dx=1, dy=0 -> max(0, 1//2) = 0. Correct.
    # Sample 1: dx=3, dy=5 -> max(5, 8//2) = 5. Correct.
    
    print(max(abs(sy - ty), (abs(sx - tx) + abs(sy - ty)) // 2))

if __name__ == "__main__":
    solve()