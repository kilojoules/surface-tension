```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 rectangles aligned horizontally.
    # A tile is defined by the pair (floor(i/2) if i+j even else ...)
    # Actually, the rule is: 
    # For a fixed j, the squares A_{i,j} are grouped in pairs.
    # If j is even: (0,1), (2,3), (4,5) ... are tiles.
    # If j is odd: (-1,0), (1,2), (3,4) ... are tiles.
    
    # Let's define the tile ID for a cell (i, j):
    # If j is even, the tile is {(2k, j), (2k+1, j)}. The ID is (k, j).
    # If j is odd, the tile is {(2k-1, j), (2k, j)}. The ID is (k, j).
    
    # To simplify:
    # For any (i, j), the tile it belongs to can be uniquely identified by:
    # If j % 2 == 0: tile_x = i // 2, tile_y = j
    # If j % 2 == 1: tile_x = (i + 1) // 2, tile_y = j
    
    # However, we can observe that the cost is simply the Manhattan distance 
    # in a transformed coordinate system.
    # Let's map (i, j) to a coordinate (u, v) such that moving between 
    # adjacent tiles costs 1.
    
    # The tiles are 2x1. 
    # In row j, tiles are at x-intervals [0,2), [2,4)... if j even
    # and x-intervals [-1,1), [1,3)... if j odd.
    
    # Let's normalize the x-coordinate based on the parity of j.
    # If j is even, x' = x // 2
    # If j is odd, x' = (x + 1) // 2
    
    # The distance between (sx, sy) and (tx, ty) is the minimum tolls.
    # Each move in y direction always enters a new tile.
    # Each move in x direction might enter a new tile.
    
    # The cost is |sy - ty| + distance_in_x.
    # The x-distance depends on the starting and ending tiles.
    # Let ux = sx // 2 if sy % 2 == 0 else (sx + 1) // 2
    # Let vx = tx // 2 if ty % 2 == 0 else (tx + 1) // 2
    
    # The total cost is |sy - ty| + |ux - vx|.
    # But wait, if we move in y, we might change the "column" we are in.
    # Actually, the problem is simpler:
    # The cost is the distance in the graph where nodes are tiles.
    # Two tiles are connected if they share an edge.
    # The distance is simply |sy - ty| + |ux - vx|.
    
    # Let's verify with Sample 1: (5, 0) to (2, 5)
    # sx=5, sy=0 -> ux = 5 // 2 = 2
    # tx=2, ty=5 -> vx = (2 + 1) // 2 = 1
    # Cost = |0 - 5| + |2 - 1| = 5 + 1 = 6. 
    # Sample output says 5. Why?
    # Because we can move to a different x-column using a y-move.
    # If we are at (5, 0), we are in tile (2, 0).
    # We can move to (4, 0) for free (same tile).
    # Then move to (4, 1). This is tile (2, 1).
    # Then move to (3, 1) for free.
    # Then move to (3, 2). This is tile (1, 2).
    # The x-coordinate of the tile changes when we move in y.
    
    # Correct logic:
    # The cost is the distance in the dual graph.
    # The tiles are the nodes.
    # The distance is the minimum number of tiles entered.
    # This is equivalent to the distance in a grid where some edges are weight 0.
    # Specifically, the edge between (i, j) and (i+1, j) is weight 0 if i+j is even.
    
    # This is a shortest path problem on a graph.
    # Since the coordinates are huge, we need a closed form.
    # The cost is simply:
    # abs(sy - ty) + (abs(sx - tx) + 1) // 2
    # Let's check Sample 1: abs(0-5) + (abs(5-2)+1)//2 = 5 + 2 = 7. Still not 5.
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    
    # If he moves from (5, 0) to (4, 0), he is in the same tile.
    # From (4, 0) to (4, 1), he enters a new tile.
    # From (4, 1) to (3, 1), he is in the same tile.
    # From (3, 1) to (3, 2), he enters a new tile.
    # From (3, 2) to (2, 2), he is in the same tile.
    # From (2, 2) to (2, 3), he enters a new tile.
    # From (2, 3) to (1, 3), he is in the same tile.
    # From (1, 3) to (1, 4), he enters a new tile.
    # From (1, 4) to (0, 4), he is in the same tile.
    # From (0, 4) to (0, 5), he enters a new tile.
    # Total tolls: 5.
    
    # The pattern is: he can move 1 unit in x for free every time he moves 1 unit in y.
    # The cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2)
    # Wait, that's for a different problem.
    # Let's use the property: he can change his x-tile index by 1 for every y-move.
    # The cost is abs(sy - ty) + max(0, (abs(sx - tx) + 1) // 2 - abs(sy - ty))
    # Which simplifies to max(abs(sy - ty), (abs(sx - tx) + 1) // 2)
    # Let's check Sample 1: max(5, (3+1)//2) = 5. Correct.
    # Sample 2: (3, 1) to (4, 1). max(0, (1+1)//2) = 1. 
    # But Sample 2 output is 0. Why?
    # Because (3, 1) and (4, 1) are in the same tile!
    # i=3, j=1 -> i+j = 4 (even). So A_{3,1} and A_{4,1} are one tile.
    # My formula (abs(sx-tx)+1)//2 is for when they are NOT in the same tile.
    
    # Let's use the tile IDs:
    # ux = sx // 2 if sy % 2 == 0 else (sx + 1) // 2
    # vx = tx // 2 if ty % 2 == 0 else (tx + 1) // 2
    # The cost is abs(sy - ty) + abs(ux - vx)
    # But he can "diagonal" move.
    # The distance is actually the L1 distance in the tile-coordinate system.
    # However, the tile-coordinate system is shifted every row.
    # The distance is simply abs(sy - ty) + abs(ux - vx)
    # Let's check Sample 1: ux = 5//2 = 2, vx = (2+1)//2 = 1. 
    # Cost = abs(0-5) + abs(2-1) = 5 + 1 = 6. Still 6.
    
    # Let's reconsider: he can move in x for free if he is in a 2x1 tile.
    # He can move 2 units in x for the cost of 1 (by moving into the next tile).
    # But he can also move 1 unit in x for free by moving in y.
    # The minimum cost is actually:
    # If we are at (sx, sy) and want to go to (tx, ty):
    # We must pay for every y-step: abs(sy - ty).
    # We must pay for every x-step that doesn't coincide with a y-step.
    # Each y-step allows us to move 1 unit in x for free (by picking the right tile).
    # The number of x-steps is abs(sx - tx).
    # We can cover abs(sy - ty) of these x-steps using the y-moves.
    # The remaining x-steps are abs(sx - tx) - abs(sy - ty).
    # These remaining steps cost 1 for every 2 units.
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) - abs(sy - ty) + 1) // 2)
    
    # Check Sample 1: 5 + max(0, (3 - 5 + 1) // 2) = 5 + 0 = 5. Correct.
    # Check Sample 2: 0 + max(0, (1 - 0 + 1) // 2) = 1. Still 1.
    # Wait, Sample 2: (3, 1) and (4, 1). i=3, j=1. i+j=4 (even).
    # They are in the same tile. The cost should be 0.
    # My formula gives 1. The issue is the "starting tile is free".
    # If they are in the same tile, cost is 0.
    # If they are in different tiles, he must pay to enter the first new tile.
    
    # Let's use the tile IDs again.
    # Tile ID: (ux, uy)
    # ux = sx // 2 if sy % 2 == 0 else (sx + 1) // 2
    # vx = tx // 2 if ty % 2 == 0 else (tx + 1) // 2
    # The distance is abs(sy - ty) + abs(ux - vx).
    # But he can move diagonally in the tile-grid.
    # A move in y changes uy by 1 and can change ux by 0 or 1.
    # This is like a grid where you can move (0, 1) or (1, 1) for cost 1.
    # The distance is max(abs(sy - ty), abs(ux - vx)).
    
    # Check Sample 1: max(5, abs(2 - 1)) = 5. Correct.
    # Check Sample 2: max(0, abs((3+1)//2 - (4+1)//2)) = max(0, abs(2 - 2)) = 0. Correct.
    # Check Sample 3: 
    # sx = 2552608206527595, sy = 5411232866732612
    # tx = 771856005518028, ty = 7206210729152763
    # ux = 2552608206527595 // 2 = 1276304103263797
    # vx = (771856005518028 + 1) // 2 = 385928002759014
    # abs(sy - ty) = abs(5411232866732612 - 7206210729152763) = 17949778