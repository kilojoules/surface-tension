```python
import sys

def solve():
    # Read input from standard input
    try:
        line = sys.stdin.read().split()
        if not line:
            return
        sx, sy, tx, ty = map(int, line)
    except EOFError:
        return

    # The problem describes a grid of 2x1 tiles.
    # Rule: If i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed row j:
    # If j is even: tiles are {(0,j), (1,j)}, {(2,j), (3,j)}, ...
    # If j is odd: tiles are {(-1,j), (0,j)}, {(1,j), (2,j)}, ...
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # For a square A_{i,j}, it belongs to a tile.
    # If j is even: tile index is ( (i // 2), j )
    # If j is odd: tile index is ( (i - 1) // 2, j ) 
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's re-evaluate:
    # Row j:
    # If j is even: i=0,1 are together; i=2,3 are together... (i // 2)
    # If j is odd: i=1,2 are together; i=3,4 are together... ((i-1) // 2)
    # Actually, the rule "i+j is even" means:
    # For j=0: i=0,2,4... are starts of tiles. A_{0,0} & A_{1,0} are one tile.
    # For j=1: i=1,3,5... are starts of tiles. A_{1,1} & A_{2,1} are one tile.
    # In general, in row j, the tile boundaries are at x = k*2 + (j % 2).
    
    # Let's map (i, j) to tile coordinates (U, V):
    # V = j
    # If j % 2 == 0: U = i // 2
    # If j % 2 == 1: U = (i - 1) // 2
    
    # However, the cost is based on entering a tile.
    # Moving within a tile is free.
    # Moving from tile A to tile B costs 1.
    # This is equivalent to the Manhattan distance in the tile-coordinate system,
    # but we must be careful because the "U" coordinate shifts based on "V".
    
    # Let's use the property that the cost to move between (sx, sy) and (tx, ty)
    # is related to the number of tile boundaries crossed.
    # Vertical boundaries are at x = k*2 + (j % 2).
    # Horizontal boundaries are always at y = k.
    
    # The cost is the minimum number of tiles entered.
    # This is a shortest path problem on a graph where nodes are tiles.
    # The distance between tile (u1, v1) and (u2, v2) is |u1 - u2| + |v1 - v2|.
    # But wait, the "U" coordinate depends on "V".
    # Let's use the transformation:
    # A square (i, j) belongs to tile ( (i + (j%2)) // 2, j )
    
    u_s = (sx + (sy % 2)) // 2
    v_s = sy
    u_t = (tx + (ty % 2)) // 2
    v_t = ty
    
    # The distance between (u_s, v_s) and (u_t, v_t) in a grid where 
    # you can move to adjacent tiles.
    # From (u, v), you can move to:
    # (u-1, v), (u+1, v) -> cost 1
    # (u, v-1), (u, v+1) -> cost 1
    # BUT, moving from (u, v) to (u, v+1) is only possible if the square 
    # (i, v) and (i, v+1) are in different tiles.
    # Actually, the problem is simpler: 
    # You pay 1 every time you enter a NEW tile.
    # The starting tile is already "entered" (or rather, you start inside it),
    # but the problem says "Each time he enters a tile, he pays a toll of 1."
    # This usually means the starting tile is free, and every subsequent tile costs 1.
    # This is exactly the Manhattan distance between the tiles.
    
    # Let's check the distance:
    # The distance between tile (u_s, v_s) and (u_t, v_t) is:
    # We want to minimize |u_s - u| + |v_s - v| + |u - u_t| + |v - v_t| 
    # is not correct because u depends on v.
    
    # Let's use the coordinate system:
    # Tile ID for (i, j) is ( (i + (j%2)) // 2, j )
    # To get from (u_s, v_s) to (u_t, v_t):
    # You can moveに v_s -> v_t (cost |v_s - v_t|)
    # And you can move u_s -> u_t (cost |u_s - u_t|)
    # However, you can change your 'u' for free when changing 'v' 
    # if the tiles overlap.
    # Specifically, square (i, j) and (i, j+1) are in tiles:
    # T1 = ((i + (j%2)) // 2, j)
    # T2 = ((i + ((j+1)%2)) // 2, j+1)
    # Note that (i + (j%2)) // 2 and (i + ((j+1)%2)) // 2 can differ by at most 1.
    
    # Let's simplify:
    # The cost is max(|u_s - u_t|, |v_s - v_t|) if we could move diagonally? No.
    # The cost is actually:
    # If we move from v_s to v_t, we must pass through |v_s - v_t| boundaries.
    # Each such move costs 1.
    # While moving vertically, we can potentially change our "u" coordinate.
    # In each step v -> v+1, the tile index u can stay the same or change by 1.
    # So in |v_s - v_t| steps, we can cover a horizontal distance of |v_s - v_t|.
    # The remaining horizontal distance is max(0, |u_s - u_t| - |v_s - v_t|).
    # But wait, the "u" change during a "v" move is only if we are at the boundary.
    # Let's re-evaluate.
    # In one step (cost 1), we can move from (u, v) to (u, v+1) or (u+/-1, v).
    # But we can also move from (u, v) to (u', v+1) if the tiles share a boundary.
    # Tile (u, v) covers x in [2u - (v%2), 2u + 1 - (v%2)]
    # Tile (u', v+1) covers x in [2u' - ((v+1)%2), 2u' + 1 - ((v+1)%2)]
    # They share a boundary if the intervals overlap.
    # The interval for (u, v) is [2u - (v%2), 2u + 1 - (v%2)]
    # The interval for (u', v+1) is [2u' - (1 - v%2), 2u' + 1 - (1 - v%2)]
    # These overlap if 2u' - (1 - v%2) <= 2u + 1 - (v%2) AND 2u - (v%2) <= 2u' + 1 - (1 - v%2)
    # This simplifies to: 2u' - 2u <= 1 and 2u - 2u' <= 1
    # Which means u' = u.
    # So you cannot change u for free while changing v.
    # Wait, the sample 1: (5,0) to (2,5).
    # s: i=5, j=0 -> u_s = (5+0)//2 = 2, v_s = 0
    # t: i=2, j=5 -> u_t = (2+1)//2 = 1, v_t = 5
    # Manhattan distance: |2-1| + |0-5| = 1 + 5 = 6.
    # But sample output is 5. Why?
    # Because you can move from tile (u, v) to (u, v+1) and that costs 1.
    # The starting tile is free.
    # So the cost is |u_s - u_t| + |v_s - v_t|.
    # For sample 1: |2-1| + |0-5| = 6. Still not 5.
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is already entered.
    # Move 1: Left 1. (5,0) is in tile ((5+0)//2, 0) = (2,0).
    # (4,0) is in tile ((4+0)//2, 0) = (2,0). Same tile! Cost 0.
    # Move 2: Up 1. (4,1) is in tile ((4+1)//2, 1) = (2,1). New tile. Cost 1.
    # Move 3: Left 1. (3,1) is in tile ((3+1)//2, 1) = (2,1). Same tile! Cost 0.
    # Move 4: Up 3. (3,2), (3,3), (3,4).
    # (3,2) is in tile ((3+0)//2, 2) = (1,2). Cost 1.
    # (3,3) is in tile ((3+1)//2, 3) = (2,3). Cost 1.
    # (3,4) is in tile ((3+0)//2, 4) = (1,4). Cost 1.
    # Move 5: Left 1. (2,4) is in tile ((2+0)//2, 4) = (1,4). Same tile! Cost 0.
    # Move 6: Up 1. (2,5) is in tile ((2+1)//2, 5) = (1,5). New tile. Cost 1.
    # Total cost: 1 + 1 + 1 + 1 + 1 = 5.
    
    # Let's trace the tiles: (2,0) -> (2,1) -> (1,2) -> (2,3) -> (1,4) -> (1,5)
    # The sequence of tiles is (u_0, v_0), (u_1, v_1), ...
    # In each step, we can change u by 1 or v by 1.
    # But notice that when we move from v to v+1, the tile index u can change.
    # Square (i, v) is in tile ((i + (v%2)) // 2, v).
    # Square (i, v+1) is in tile ((i + ((v+1)%2)) // 2, v+1).
    # Let f(i, v) = (i + (v%2)) // 2.
    # If v is even, f(i, v) = i // 2. Then f(i, v+1) = (i+1) // 2.
    # If v is odd, f(i, v) = (i+1) // 2. Then f(i, v+1) = i // 2.
    # Notice that for any i, {f(i, v), f(i, v+1)} is always {k, k} or {k, k+1} or {k, k-1}.
    # Specifically, if i is even, f(i, 0)=i/2, f(i, 1)=(i+1)//2 = i/2. (Same u!)
    # If i is odd, f(i, 0)=i//2,