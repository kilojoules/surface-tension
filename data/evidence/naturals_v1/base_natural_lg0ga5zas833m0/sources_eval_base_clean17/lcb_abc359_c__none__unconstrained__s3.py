import sys

def solve():
    """
    The problem describes a grid of 2x1 tiles.
    A square A_{i,j} is part of a tile.
    If i+j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    This means:
    For a fixed j:
    - If j is even: (0,j) & (1,j) are together, (2,j) & (3,j) are together...
    - If j is odd: (-1,j) & (0,j) are together, (1,j) & (2,j) are together...
    
    Essentially, for a fixed row j, the tiles are horizontal segments of length 2.
    The boundary between tiles in row j is at x = k where (k-1)+j is even (i.e., k+j is odd).
    
    Wait, the rule says: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    Let's analyze row j:
    If j is even: i=0, 2, 4... are starts of tiles. Tiles are [0,2], [2,4]...
    If j is odd: i=1, 3, 5... are starts of tiles. Tiles are [1,3], [3,5]... and also [-1,1].
    
    Takahashi starts at (Sx+0.5, Sy+0.5) and ends at (Tx+0.5, Ty+0.5).
    Moving within a tile costs 0. Entering a new tile costs 1.
    This is equivalent to finding the distance in a graph where nodes are tiles.
    
    Let's transform the coordinates to a simpler grid.
    A tile can be uniquely identified by (row j, tile_index k).
    In row j, a square A_{i,j} belongs to tile (j, (i - (j % 2)) // 2).
    Let's call this coordinate (X, Y) where Y = j and X = (i - (j % 2)) // 2.
    
    Wait, let's simplify:
    A square (i, j) is in tile:
    If j is even: Tile is {(i, j), (i+1, j)} if i is even.
    If j is odd: Tile is {(i, j), (i+1, j)} if i is odd.
    
    Let's map (i, j) to a "tile coordinate" (u, v).
    v = j
    u = (i + (j % 2)) // 2
    
    Now let's see the movement:
    1. Move horizontally (change i):
       If we move from (i, j) to (i+1, j), we might enter a new tile.
       In our (u, v) coordinates, if i and i+1 are in the same tile, u doesn't change.
       If they are in different tiles, u changes by 1.
    2. Move vertically (change j):
       If we move from (i, j) to (i, j+1), we always enter a new tile because tiles are only 2x1 (horizontal).
       So moving from v to v+1 always costs 1.
    
    Wait, the cost is "each time he enters a tile".
    Starting tile is free.
    Moving from (u1, v1) to (u2, v2):
    The distance is essentially the L1 distance in the (u, v) space, but we must be careful.
    
    Let's re-evaluate:
    From (u, v), he can move to:
    - (u+1, v) or (u-1, v) by moving horizontally. Cost: 1.
    - (u, v+1) or (u, v-1) by moving vertically.
      However, moving from (i, j) to (i, j+1) might land him in the same tile as (i+1, j+1) or (i-1, j+1).
      Actually, since tiles are only 2x1, any vertical move always enters a new tile.
      But can he move to (u+1, v+1) or (u-1, v+1) in one vertical move?
      No, a vertical move is (x, y) -> (x, y+n). He stays in the same x-column.
      A vertical move from (i, j) to (i, j+1) enters the tile containing (i, j+1).
      The tile containing (i, j+1) is (u', v+1) where u' = (i + ((j+1)%2)) // 2.
      
      Let's check the relationship between u and u':
      u = (i + (j%2)) // 2
      u' = (i + ((j+1)%2)) // 2
      If j is even: u = (i)//2, u' = (i+1)//2.
      If j is odd: u = (i+1)//2, u' = (i)//2.
      In both cases, u' is either u or u +/- 1.
      Specifically, if i is even:
      j even: u = i/2, u' = (i+1)//2 = i/2. (u' = u)
      j odd: u = (i+1)//2, u' = i//2. (u' = u - 1 if i is even? No, let's re-calc)
      
      Let's use a concrete example: i=0.
      j=0 (even): u = (0+0)//2 = 0.
      j=1 (odd): u' = (0+1)//2 = 0.
      i=1.
      j=0 (even): u = (1+0)//2 = 0.
      j=1 (odd): u' = (1+1)//2 = 1.
      
      So from (u, v), a vertical move to v+1 lands him in either u or u +/- 1.
      This is like a grid where you can move to (u, v+1), (u+1, v+1), or (u-1, v+1).
      The cost to move to any of these is 1.
      
      This is the distance in a grid where you can move to 8-neighbors (Chebyshev distance) but only in certain ways?
      Actually, it's simpler:
      The distance between (u1, v1) and (u2, v2) is max(|u1-u2|, |v1-v2|).
      Wait, let's verify.
      If |v1-v2| is the dominant distance, he can move vertically and adjust u by 1 each step.
      If |u1-u2| is dominant, he can move horizontally.
      
      Let's re-verify the cost:
      - Horizontal move: (u, v) -> (u+1, v). Cost 1.
      - Vertical move: (i, j) -> (i, j+1). He enters tile (u', v+1). Cost 1.
      As shown, u' can be u or u +/- 1.
      So in one vertical move, he can change v by 1 and u by -1, 0, or 1.
      This is exactly the Chebyshev distance: max(|u1-u2|, |v1-v2|).
      
      Wait, the starting tile is free.
      The cost is the number of *new* tiles entered.
      Distance = max(|u1-u2|, |v1-v2|).
      
      Let's test Sample 1:
      S: (5, 0) -> i=5, j=0. u = (5 + 0)//2 = 2, v = 0.
      T: (2, 5) -> i=2, j=5. u = (2 + 1)//2 = 1, v = 5.
      max(|2-1|, |0-5|) = max(1, 5) = 5. Correct.
      
      Sample 2:
      S: (3, 1) -> i=3, j=1. u = (3+1)//2 = 2, v = 1.
      T: (4, 1) -> i=4, j=1. u = (4+1)//2 = 2, v = 1.
      max(0, 0) = 0. Correct.
    """
    # Reading input
    try:
        line1 = sys.stdin.readline().split()
        if not line1: return
        sx, sy = map(int, line1)
        line2 = sys.stdin.readline().split()
        if not line2: return
        tx, ty = map(int, line2)
    except ValueError:
        return

    # Transform to tile coordinates (u, v)
    # u = (i + (j % 2)) // 2
    # v = j
    
    u1 = (sx + (sy % 2)) // 2
    v1 = sy
    u2 = (tx + (ty % 2)) // 2
    v2 = ty
    
    ans = max(abs(u1 - u2), abs(v1 - v2))
    print(ans)

if __name__ == "__main__":
    solve()