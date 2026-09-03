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

    # The rule for tiling:
    # If i + j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), (4,5)... if j is even.
    # If j is odd, the pairs are (-1,0), (1,2), (3,4)...
    # Essentially, if (i + j) is even, the tile spans x from i to i+2.
    # Let's define a coordinate transformation to make the grid uniform.
    # In the original grid, moving in Y always crosses a tile boundary.
    # Moving in X might or might not cross a boundary depending on the parity of i+j.
    
    # Let's analyze the cost:
    # To move from (sx, sy) to (tx, ty):
    # The vertical distance is |sy - ty|. Each unit of vertical movement 
    # necessarily enters a new tile because tiles are 2x1 (horizontal).
    # However, we can optimize by picking X coordinates that allow us to 
    # "skip" tolls or move efficiently.
    
    # Let's redefine the coordinates to a system where tiles are aligned.
    # A tile is defined by the pair of squares it covers.
    # If i+j is even, tile is {(i,j), (i+1,j)}.
    # This is equivalent to saying: a tile is identified by (floor((i + (j%2)) / 2), j).
    # Let X' = (i + (j%2)) // 2 and Y' = j.
    # The distance in Y' is simply |sy - ty|.
    # The distance in X' is |X'_s - X'_t|.
    
    # Wait, the problem says "Each time he enters a tile, he pays a toll of 1."
    # If he is already in a tile and moves within it, cost is 0.
    # If he moves to a different tile, cost is 1.
    # This is like finding the shortest path on a graph where nodes are tiles.
    # Two tiles are adjacent if they share an edge.
    # The cost to move between adjacent tiles is 1.
    # The distance is the L1 distance in the transformed coordinate system.
    
    # Let's refine the transformation:
    # For a square (i, j), it belongs to tile T(i, j).
    # If (i + j) is even, T(i, j) = T(i + 1, j).
    # Let's map (i, j) to a tile coordinate (U, V).
    # V = j
    # If j is even: i is grouped as (0,1), (2,3)... so U = i // 2
    # If j is odd: i is grouped as (-1,0), (1,2)... so U = (i + 1) // 2
    # This can be written as: U = (i + (j % 2)) // 2
    
    # The distance between tile (Us, Vs) and (Ut, Vt) in a grid where you can 
    # move to any adjacent tile is |Us - Ut| + |Vs - Vt|.
    # However, the problem allows moving any distance n in one direction.
    # "Choose a direction... and a positive integer n. Move n units."
    # This means if he moves vertically, he crosses |sy - ty| boundaries.
    # If he moves horizontally, he crosses |Us - Ut| boundaries.
    # But he can alternate.
    
    # Let's re-evaluate:
    # He starts in tile (Us, Vs) and ends in (Ut, Vt).
    # Each move in a direction costs 1 per tile entered.
    # A move of n units in Y enters n new tiles.
    # A move of n units in X enters some number of tiles.
    # Because he can move any n, he can effectively move to any (X', Y') 
    # and the cost is the number of tile boundaries crossed.
    # The minimum cost to get from (Us, Vs) to (Ut, Vt) is |Us - Ut| + |Vs - Vt|.
    # But wait, he starts inside a tile. The first tile is already "entered".
    # The cost is the number of NEW tiles entered.
    # Distance = |Us - Ut| + |Vs - Vt|.
    
    # Let's check Sample 1: (5, 0) to (2, 5)
    # S: i=5, j=0. U_s = (5 + 0)//2 = 2. V_s = 0.
    # T: i=2, j=5. U_t = (2 + 1)//2 = 1. V_t = 5.
    # Cost = |2 - 1| + |0 - 5| = 1 + 5 = 6.
    # Sample 1 output is 5. Why?
    # Because he can move to a position where the X-boundary is not crossed.
    # If he is at (5.5, 0.5), he is in tile U=2, V=0.
    # He can move to (4.5, 0.5), which is tile U=2, V=0 (since i=4, j=0 -> U=(4+0)//2 = 2).
    # Then move up to (4.5, 5.5).
    # Let's trace:
    # (5.5, 0.5) [U=2, V=0]
    # Move left 1 -> (4.5, 0.5) [U=2, V=0] - Cost 0
    # Move up 1 -> (4.5, 1.5) [U=(4+1)//2=2, V=1] - Cost 1
    # Move left 1 -> (3.5, 1.5) [U=(3+1)//2=2, V=1] - Cost 0
    # Move up 4 -> (3.5, 5.5) [U=(3+5)//2=4? No.]
    # Let's use the formula:
    # Start: (5, 0) -> Us = (5 + 0)//2 = 2, Vs = 0
    # End: (2, 5) -> Ut = (2 + 1)//2 = 1, Vt = 5
    # The cost is |Us - Ut| + |Vs - Vt|, but we can pick whether to 
    # use the "left" or "right" side of the tile to minimize.
    # Actually, the simplest way to think about this is:
    # The cost is |Vs - Vt| + (cost to change U).
    # To change U from Us to Ut, he might need to move horizontally.
    # But he can change his U coordinate "for free" by moving vertically 
    # if the tile boundaries shift.
    
    # Let's use the property:
    # Cost = |Vs - Vt| + max(0, |Us - Ut| - (some value))
    # Actually, the most reliable way to solve this is to recognize that
    # the distance is |Vs - Vt| + |Us - Ut|, but you can potentially 
    # reduce the |Us - Ut| part if you use the vertical movement to 
    # "shift" your U coordinate.
    # However, the parity of V changes every step.
    # If you move from V to V+1, the U coordinate of the tile containing 
    # the same X coordinate might change.
    # Let's use the coordinate transformation:
    # X_new = i + (j % 2)
    # Y_new = 2 * j
    # This is not quite right.
    
    # Correct approach:
    # The distance is |Vs - Vt| + |Us - Ut|.
    # But you can move diagonally in the (U, V) space by moving 
    # 1 unit in X and 1 unit in Y in the original space.
    # Wait, the sample 1: |2-1| + |0-5| = 6, but answer is 5.
    # This happens if you can move from (Us, Vs) to (Ut, Vt) 
    # using a path that "saves" a step.
    # If you move from (U, V) to (U, V+1), the X-coordinate i 
    # might belong to the same tile U in both V and V+1.
    # Example: i=4, j=0 -> U=(4+0)//2 = 2.
    # i=4, j=1 -> U=(4+1)//2 = 2.
    # So moving from (4.5, 0.5) to (4.5, 1.5) costs 1 (for the V change)
    # and you are still in U=2.
    # But if you were at i=5, j=0 (U=2), and you move to j=1, 
    # you are now at i=5, j=1 (U=(5+1)//2 = 3).
    # So you can change your U coordinate by 1 just by moving vertically!
    
    # Let's formalize:
    # You are at (Us, Vs). You want to reach (Ut, Vt).
    # In one vertical move (cost 1), you can move from (U, V) to (U, V+1) 
    # OR (U+1, V+1) OR (U-1, V+1) depending on the X coordinate.
    # Specifically, for a fixed X, the tile index is U = (X + (V%2)) // 2.
    # When V changes to V+1, U becomes (X + ((V+1)%2)) // 2.
    # The difference is either 0 or 1.
    # This means for every vertical step, you can potentially change U by 1 for free.
    # The total cost is |Vs - Vt| + max(0, |Us - Ut| - |Vs - Vt|).
    # This simplifies to max(|Vs - Vt|, |Us - Ut| + |Vs - Vt| - |Vs - Vt|) 
    # which is just max(|Vs - Vt|, |Us - Ut|).
    # Wait, that's not right. Let's re-calculate.
    # Cost = |Vs - Vt| + max(0, |Us - Ut| - |Vs - Vt|) 
    # is actually just |Us - Ut| if |Us - Ut| > |Vs - Vt| else |Vs - Vt|.
    # Which is max(|Us - Ut|, |Vs - Vt|).
    # Let's check Sample 1: Us=2, Vs=0, Ut=1, Vt=5.
    # max(|2-1|, |0-5|) = max(1, 5) = 5. Correct!
    # Sample 2: (3, 1) to (4, 1)
    # Us = (3+1)//2 = 2, Vs = 1
    # Ut = (4+1)//2 = 2, Vt = 1
    # max(|2-2|, |1-1|) = 0. Correct!
    
    # Final logic:
    # Us = (sx + (sy % 2)) // 2
    # Vs = sy
    # Ut = (tx + (ty % 2)) // 2
    # Vt = ty
    # Result = max(abs(Us - Ut), abs(Vs - Vt))
    
    us = (sx + (sy % 2)) // 2
    vs = sy
    ut = (tx + (ty % 2)) // 2
    vt = ty
    
    print(max(abs(us - ut), abs(vs - vt)))

if __name__ == "__main__":
    solve()