```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), (4,5)... if j is even
    # and pairs (1,2), (3,4), (5,6)... if j is odd.
    
    # Let's define a coordinate transformation to a grid where each unit is a tile.
    # For a cell (x, y), it belongs to a tile.
    # If y is even, the tile index in x is floor(x/2).
    # If y is odd, the tile index in x is floor((x+1)/2).
    # However, the vertical movement is simpler: every change in y enters a new tile.
    
    # Let's determine the tile coordinates (U, V) for a cell (x, y).
    # V = y
    # If y % 2 == 0: U = x // 2
    # If y % 2 == 1: U = (x + 1) // 2
    
    # But wait, the cost is the number of tiles ENTERED.
    # If we are already in a tile, the first tile doesn't count.
    # The cost is the distance in the tile-graph.
    # In the tile-graph, moving from (U, V) to (U', V') costs:
    # cost = abs(V - V') + (abs(U - U') if we aren't already in the same tile)
    
    # Let's refine the tile coordinates:
    # For cell (x, y):
    # The tile ID is ( (x // 2) if (x + y) % 2 == 0 else ((x+1) // 2), y )
    # Actually, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j:
    # If j is even: tiles are {0,1}, {2,3}, {4,5} ... -> tile_x = x // 2
    # If j is odd:  tiles are {-1,0}, {1,2}, {3,4} ... -> tile_x = (x + 1) // 2
    
    # Let's use the logic:
    # Start tile: (sx // 2 if sy % 2 == 0 else (sx + 1) // 2, sy)
    # End tile: (tx // 2 if ty % 2 == 0 else (tx + 1) // 2, ty)
    
    # The distance between tiles (U1, V1) and (U2, V2) in this specific grid:
    # You can move V by 1 (cost 1).
    # You can move U by 1 (cost 1), but only if you are on a row that allows that 
    # specific tile boundary.
    # Actually, the simplest way to view this is:
    # To change U by 1, you must be in a tile that spans the boundary.
    # The cost is simply abs(V1 - V2) + abs(U1 - U2).
    # But there is a catch: if you move vertically, you might land in a tile 
    # that already covers the X-coordinate you need.
    
    # Correct logic for this specific tiling:
    # The distance is abs(sy - ty) + abs(U1 - U2)
    # UNLESS the vertical movement allows you to "skip" a horizontal move.
    # If you move from sy to ty, you cross abs(sy - ty) boundaries.
    # One of those rows might have a tile that covers both sx and tx.
    # That happens if abs(U1 - U2) == 0.
    
    # Let's use the coordinate system:
    # U(x, y) = x // 2 if y % 2 == 0 else (x + 1) // 2
    # Cost = abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # However, if sy == ty and the two points are in the same tile, cost is 0.
    # The formula abs(sy - ty) + abs(U1 - U2) covers this.
    # Wait, if sy != ty, can we reduce the cost?
    # If we move from (U1, V1) to (U2, V2), the cost is abs(V1 - V2) + abs(U1 - U2).
    # But if we change V, we might change our U coordinate for free.
    # Example: sx=5, sy=0 -> U1 = 5//2 = 2. tx=2, ty=5 -> U2 = (2+1)//2 = 1.
    # Cost = abs(0-5) + abs(2-1) = 5 + 1 = 6. 
    # But Sample 1 says 5. Why?
    # Because at some y, the tile might cover both x=5 and x=2? No, tiles are 2x1.
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are one tile."
    # For j=0: (0,1), (2,3), (4,5) are tiles. x=5 is in tile {4,5}.
    # For j=1: (1,2), (3,4), (5,6) are tiles. x=5 is in tile {5,6}.
    # For j=2: (0,1), (2,3), (4,5) are tiles. x=2 is in tile {2,3}.
    # For j=3: (1,2), (3,4), (5,6) are tiles. x=2 is in tile {1,2}.
    
    # Let's use the property:
    # The cost is abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # BUT, if we move vertically, we can pick the "best" U for the destination.
    # The cost is actually:
    # min(
    #   abs(sy - ty) + abs(U(sx, sy) - U(tx, ty)),
    #   abs(sy - ty) + abs(U(sx, sy) - U(tx, ty-1)) if ty != sy,
    #   ...
    # )
    # Actually, the most robust way:
    # The cost is abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # But we can potentially save 1 if we can reach the target tile by just 
    # moving vertically from a tile that was already "closer" in U.
    # The correct formula for this problem is:
    # cost = abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # Then, if sy != ty, we can check if moving to ty-1 or ty+1 first 
    # (which we do anyway) allows a cheaper U-transition.
    # Actually, the simplest observation:
    # You pay for every vertical step. You pay for every horizontal tile boundary crossed.
    # The number of horizontal boundaries between U1 and U2 is abs(U1 - U2).
    # Total = abs(sy - ty) + abs(U(sx, sy) - U(tx, ty)).
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5.
    # U(5, 0) = 5 // 2 = 2.
    # U(2, 5) = (2 + 1) // 2 = 1.
    # Cost = abs(0 - 5) + abs(2 - 1) = 5 + 1 = 6.
    # Still 6. Sample says 5. What's wrong?
    # "Move left by 1. Pay a toll of 0." -> (5,0) to (4,0). Both are in tile {4,5}.
    # "Move up by 1. Pay a toll of 1." -> (4,0) to (4,1).
    # "Move left by 1. Pay a toll of 0." -> (4,1) to (3,1). Both are in tile {3,4}.
    # This means he is changing his U coordinate by moving vertically!
    # At y=0, x=4 is U=2. At y=1, x=4 is U=2 (since (4+1)//2 = 2).
    # Wait, at y=1, x=3 is also U=2 (since (3+1)//2 = 2).
    # So by moving from (4,0) to (4,1), he is now in a tile that covers x=3.
    # This means the U-coordinate can change based on y.
    # The cost is simply the distance in the graph where nodes are tiles.
    # Two tiles are connected if they share a boundary.
    # The distance is abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # BUT, you can move diagonally in the (U, V) space? 
    # No, but you can change U for "free" if the tile at (U, V) and (U', V+1) 
    # share a boundary.
    # Actually, the distance is simply:
    # abs(sy - ty) + abs(U(sx, sy) - U(tx, ty)) 
    # MINUS 1 if (sy != ty) and (the parity of the tiles allows a "shortcut").
    # The shortcut is: if you move from V to V+1, you might enter a tile 
    # that is already at U2.
    # The correct formula is:
    # ans = abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # if sy != ty:
    #    ans = min(ans, abs(sy - ty) + abs(U(sx, sy) - U(tx, ty)) - 1) 
    #    # This is not quite right.
    
    # Let's use the property:
    # The distance is abs(sy - ty) + abs(U(sx, sy) - U(tx, ty))
    # But if sy != ty, you can potentially save 1 if the tiles "overlap" 
    # in a way that one vertical move covers the horizontal distance.
    # Specifically, if abs(U1 - U2) <= 1, the cost is just abs(sy - ty).
    # Wait, if U1 == U2, cost is abs(sy - ty).
    # If abs(U1 - U2) == 1, cost is abs(sy - ty) + 1.
    # But if sy != ty, you can move V first, then U.
    # If you are at (U1, V1) and want to go to (U2, V2),
    # you can move to (U1, V2) in abs(V1 - V2) steps.
    # Then from (U1, V2) to (U2, V2) in abs(U1 - U2) steps.
    # Total = abs(V1 - V2) + abs(U1 - U2).
    # However, you can also move to (U1, V2-1), then to (U2, V2-1), then to (U2, V2).
    # The only way to save is if the tile at (U1, V) and (U2, V+1) are the same.
    # But tiles are 2x1, they can't be the same if U1 != U2.
    # Let's re-examine Sample 1: U1=2, U2=1, V1=0, V2=5.
    # Cost = 5 + 1 = 6. Sample says 5.
    # The only way to get 5 is if the cost is just abs(sy - ty).
    # That happens if abs(U1 - U2) <= 1 and sy != ty? 
    # No, that's not it.
    # Let's look at the coordinates again.
    # (5,0) is in tile U=2, V=0.
    # (2,5) is in tile U=1, V=5.
    # If he moves (5,0) -> (4,0) [same tile], then (4,0) -> (4,1) [new tile],
    # then (4,1)