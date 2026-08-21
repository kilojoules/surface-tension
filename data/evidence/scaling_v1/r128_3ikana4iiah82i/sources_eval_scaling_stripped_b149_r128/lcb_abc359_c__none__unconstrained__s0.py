```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), etc. if j is even.
    # If j is odd, the tiles are pairs (1,2), (3,4), etc.
    
    # Let's define a coordinate transformation to a space where tiles are 2x1 blocks.
    # In the original space, a tile is {(i,j), (i+1,j)} if i+j is even.
    # This is equivalent to saying a tile is the set of cells {(i,j), (i',j)} 
    # where floor((i+j)/2) == floor((i'+j)/2).
    
    # Let u = i + j and v = i - j.
    # However, a simpler approach is to recognize that the cost to move 
    # between two cells (sx, sy) and (tx, ty) depends on the number of 
    # tile boundaries crossed.
    
    # For a fixed j, the boundary between cell i and i+1 is a tile boundary 
    # if i+j is odd.
    # For a fixed i, the boundary between cell j and j+1 is always a tile boundary 
    # because tiles are only 2x1 (horizontal).
    
    # Let's evaluate the cost. 
    # Moving from sx to tx at height sy:
    # We cross boundaries at x = k for k between min(sx, tx)+1 and max(sx, tx).
    # A boundary at x=k is a tile boundary if (k-1) + sy is odd.
    
    # Moving from sy to ty at width tx:
    # Every step in y crosses a tile boundary.
    
    # But we can choose the path. This looks like a shortest path problem on a graph.
    # The states can be simplified. The cost is based on the parity of i+j.
    # Let's use the property that we can move any distance n.
    # If we move horizontally, we only pay if we cross a boundary.
    # If we move vertically, we always pay.
    
    # Let's redefine: 
    # A cell (i, j) belongs to tile T(i, j).
    # If i+j is even, T(i, j) = T(i+1, j).
    # Otherwise, T(i, j) != T(i+1, j) and T(i, j) != T(i, j+1).
    
    # Let's map (i, j) to a normalized coordinate (I, J).
    # Let I = (i + (1 if (i+j)%2 != 0 else 0)) // 2 is not quite right.
    # Let's use: Tile ID for (i, j) is ( (i + (j%2)) // 2, j )
    # Wait, if i+j is even, (i,j) and (i+1,j) are same.
    # If j is even: (0,j)&(1,j), (2,j)&(3,j) ... -> Tile index is i // 2
    # If j is odd: (-1,j)&(0,j), (1,j)&(2,j) ... -> Tile index is (i-1) // 2
    # In both cases, the tile index is (i + (j % 2)) // 2.
    
    # Let U = (i + (j % 2)) // 2
    # Let V = j
    # The cost to move from (U1, V1) to (U2, V2) is:
    # We can move in U or V.
    # Moving in V always costs 1 per unit.
    # Moving in U costs 1 per unit.
    # However, we can change V to change the "grid" of U.
    # This is a shortest path on a graph where nodes are (U, V).
    # But the cost is simply the L1 distance in the (U, V) space?
    # Let's check: 
    # Start: U_s = (sx + (sy % 2)) // 2, V_s = sy
    # End: U_t = (tx + (ty % 2)) // 2, V_t = ty
    # Cost = |U_s - U_t| + |V_s - V_t|
    
    # Let's test Sample 1: 5 0 -> 2 5
    # U_s = (5 + 0) // 2 = 2, V_s = 0
    # U_t = (2 + (5 % 2)) // 2 = (2 + 1) // 2 = 1, V_t = 5
    # Cost = |2 - 1| + |0 - 5| = 1 + 5 = 6. 
    # Sample 1 output is 5. My formula is slightly off.
    
    # Re-evaluating:
    # The cost to move from (sx, sy) to (tx, ty) is the minimum of:
    # 1. Move to some height j, then move horizontally to tx, then move to ty.
    # The cost is |sy - j| + |(sx + (sy%2))//2 - (tx + (j%2))//2| + |j - ty|
    # This is minimized when j is between sy and ty.
    # If we move from sy to ty, we must pass through all heights.
    # The total cost is |sy - ty| + distance in U.
    # But we can pick the "best" height to perform the horizontal shift.
    # Actually, the cost is simply:
    # If we move from (sx, sy) to (tx, ty), we must pay |sy - ty|.
    # Additionally, we pay for the horizontal distance.
    # The horizontal distance is the number of boundaries crossed.
    # A boundary at x=k is crossed if we are at height j and (k-1)+j is odd.
    # We want to pick j in [min(sy, ty), max(sy, ty)] that minimizes the 
    # number of k's such that (k-1)+j is odd for k between sx and tx.
    
    # Let dx = |sx - tx|.
    # For a fixed j, the number of k's is:
    # If dx is even, there are dx/2 boundaries regardless of j.
    # If dx is odd, there are (dx+1)//2 or (dx-1)//2 boundaries.
    # We can pick j to get (dx-1)//2 if we can choose j such that 
    # the boundaries at the ends are not crossed.
    
    # Let's simplify:
    # The cost is |sy - ty| + (dx // 2) + (1 if dx % 2 == 1 and we are forced to cross the extra boundary)
    # We are forced to cross the extra boundary if for all j between sy and ty,
    # the parity of (min(sx, tx) + j) is such that the boundary is crossed.
    # But we can just pick j = sy or j = ty.
    # The only case we are forced to pay the extra 1 is if dx is odd AND
    # both (min(sx, tx) + sy) and (min(sx, tx) + ty) are odd.
    # Wait, if dx is odd, we can always pick j to be either sy or ty.
    # One of them will result in (dx-1)//2 boundaries if we can pick the parity.
    # Actually, if dx is odd, we can achieve (dx-1)//2 if there exists j in [sy, ty]
    # such that (min(sx, tx) + j) is even.
    # If sy == ty, we only have one choice of j.
    # If sy != ty, we can always pick j to be either sy or sy+1, so we can always 
    # get the minimum (dx-1)//2.
    
    # Let's refine:
    # 1. Vertical cost: v_cost = abs(sy - ty)
    # 2. Horizontal cost: 
    #    If sx == tx: h_cost = 0
    #    If sx != tx:
    #       dx = abs(sx - tx)
    #       If sy != ty: h_cost = dx // 2
    #       If sy == ty:
    #          # We must use height sy.
    #          # Boundaries are k = min(sx, tx)+1 ... max(sx, tx)
    #          # Boundary k is a toll if (k-1) + sy is odd.
    #          # This is a sequence of dx terms with alternating parity.
    #          # The number of odds in a sequence of length dx starting with 
    #          # parity of (min(sx, tx) + sy) is:
    #          # dx // 2 + (1 if dx % 2 == 1 and (min(sx, tx) + sy) % 2 == 1 else 0)
    #          # Wait, the first boundary is k = min(sx, tx) + 1.
    #          # It is a toll if (min(sx, tx)) + sy is odd.
    #          h_cost = dx // 2 + (1 if dx % 2 == 1 and (min(sx, tx) + sy) % 2 == 1 else 0)
    
    # Let's check Sample 1: 5 0 -> 2 5
    # v_cost = |0 - 5| = 5
    # dx = |5 - 2| = 3
    # sy != ty, so h_cost = 3 // 2 = 1
    # Total = 5 + 1 = 6. Still not 5.
    
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # If he moves from (sx, sy) to (tx, ty), he enters some number of tiles.
    # This is equivalent to: Total Tiles Visited - 1.
    # The distance is the minimum number of tiles to traverse.
    # This is a BFS problem on a graph where nodes are tiles.
    # Two tiles are adjacent if they share an edge.
    # Tile(i, j) is adjacent to:
    # 1. Tile(i, j+1) - always
    # 2. Tile(i, j-1) - always
    # 3. Tile(i+1, j) if i+j is odd (since if i+j is even, they are the same tile)
    # 4. Tile(i-1, j) if (i-1)+j is odd
    
    # Let's use the coordinate transformation:
    # Every tile can be uniquely identified by (U, V) where V = j and U = (i + (j%2)) // 2.
    # From (U, V), we can move to:
    # - (U, V+1) : cost 1
    # - (U, V-1) : cost 1
    # - (U+1, V) : cost 1 (if the boundary is crossed)
    # - (U-1, V) : cost 1 (if the boundary is crossed)
    
    # Wait, the move is "Move n units in that direction".
    # This means he can jump over many tiles in one move.
    # But he pays for EACH tile he enters.
    # So moving n units is just like moving 1 unit n times.
    # The cost is the shortest path in the tile graph.
    
    # In the tile graph:
    # Nodes: (U, V)
    # Edges:
    # (U, V) -> (U, V+1) weight 1
    # (U, V) -> (U, V-1) weight 1
    # (U, V) -> (U+1, V) weight 1
    # (U, V) -> (U-1, V) weight 1
    # This is just the L1 distance in (U, V) space!
    # Cost = |U_s - U_t| + |V_s - V_t|
    # Let's re-check Sample 1: 5 0 -> 2 5
    # U_s = (5 + (0%2)) // 2 = 5 // 2 = 2
    # V_s = 0
    # U_t = (2 + (5