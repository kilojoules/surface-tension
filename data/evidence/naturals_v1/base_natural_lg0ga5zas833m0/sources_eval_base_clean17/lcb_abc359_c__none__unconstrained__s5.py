import sys

def solve():
    """
    The problem describes a coordinate plane covered by 2x1 tiles.
    Rule: If i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    This means for a fixed row j:
    - If j is even: tiles are [0,1][2,3][4,5]... (A_{0,j} and A_{1,j} are one tile)
    - If j is odd: tiles are [1,2][3,4][5,6]... (A_{1,j} and A_{2,j} are one tile)
    
    Let's redefine the coordinates to make the grid uniform.
    A tile in row j covers x-intervals:
    - j even: [0, 2), [2, 4), ...
    - j odd: [-1, 1), [1, 3), ...
    
    Wait, the rule says: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    Let's check:
    If j=0 (even): i=0, 2, 4... (i+j even) => A_{0,0} & A_{1,0} are one tile, A_{2,0} & A_{3,0} are one tile.
    If j=1 (odd): i=1, 3, 5... (i+j even) => A_{1,1} & A_{2,1} are one tile, A_{3,1} & A_{4,1} are one tile.
    
    This is a Manhattan-like distance problem on a graph where nodes are tiles.
    Let's transform the coordinates (x, y) to (u, v) such that the tiles become 1x1 squares.
    For a cell (x, y), let its tile ID be:
    If y is even: tile_x = x // 2, tile_y = y
    If y is odd: tile_x = (x - 1) // 2, tile_y = y
    Wait, this is slightly wrong because if x=0 and y=1, (0-1)//2 = -1.
    Correct tile index for cell (x, y):
    If (x + y) % 2 == 0:
        # x is the left side of the 2x1 tile
        tile_x = (x + (1 if y % 2 != 0 else 0)) // 2
    else:
        # x is the right side of the 2x1 tile
        tile_x = (x + (1 if y % 2 != 0 else 0)) // 2
    
    Actually, a simpler way:
    A tile covers two cells (i, j) and (i+1, j) if i+j is even.
    Let's map (x, y) to a coordinate (X, Y) in a grid of tiles.
    For a cell (x, y):
    The tile index in the x-direction is:
    If y is even: X = x // 2
    If y is odd: X = (x + 1) // 2 (Wait, let's be careful)
    
    Let's use the property:
    If y is even, tiles are {(0,0),(1,0)}, {(2,0),(3,0)}...
    If y is odd, tiles are {(-1,1),(0,1)}, {(1,1),(2,1)}...
    Wait, the rule is: i+j even => A_{i,j} and A_{i+1,j} are same.
    j=0: i=0, 2, 4... => (0,0)&(1,0), (2,0)&(3,0)
    j=1: i=1, 3, 5... => (1,1)&(2,1), (3,1)&(4,1)
    
    So for cell (x, y):
    If y is even:
        X = x // 2
        # If x is odd, it's the second half of the tile.
    If y is odd:
        X = (x - 1) // 2 if x > 0 else -1 # This is confusing.
        # Let's use: X = (x + 1) // 2 if y is odd else x // 2.
        # Let's check:
        # y=0: x=0 -> 0, x=1 -> 0, x=2 -> 1, x=3 -> 1 (Correct)
        # y=1: x=0 -> 0, x=1 -> 1, x=2 -> 1, x=3 -> 2 (Wait, i+j even: 1+1=2, so A_{1,1} and A_{2,1} are one tile)
        # y=1: x=1 -> X, x=2 -> X. So (1+1)//2 = 1, (2+1)//2 = 1. (Correct)
        # y=1: x=0 -> (0+1)//2 = 0. A_{0,1} is a tile by itself? No, the plane is covered by 2x1 tiles.
        # If i+j is odd, A_{i,j} must be part of a tile. The only other option is A_{i,j} and A_{i,j+1} or A_{i,j} and A_{i,j-1}.
        # But the rules say: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
        # This means ALL tiles are horizontal 2x1 tiles.
        # Let's re-read: "The coordinate plane is covered with 2x1 tiles."
        # "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
        # This implies that if i+j is odd, A_{i,j} MUST be paired with A_{i-1,j} (since (i-1)+j is even).
        # So every single tile is a horizontal 2x1 tile.
        # For any (x, y), the tile it belongs to is:
        # If (x + y) % 2 == 0: it's the left cell of the tile {A_{x,y}, A_{x+1,y}}
        # If (x + y) % 2 != 0: it's the right cell of the tile {A_{x-1,y}, A_{x,y}}
        
        # Let's define the tile's unique ID as (tx, ty)
        # ty = y
        # tx = x // 2 if y % 2 == 0 else (x + 1) // 2
        # Wait, let's test:
        # y=0: x=0 -> 0, x=1 -> 0, x=2 -> 1, x=3 -> 1
        # y=1: x=0 -> 0, x=1 -> 1, x=2 -> 1, x=3 -> 2
        # This matches the "i+j even" rule.
        
        # Now, Takahashi moves in a grid of tiles.
        # From tile (tx, ty), he can move to:
        # 1. (tx+1, ty) and (tx-1, ty) - Moving horizontally.
        # 2. (tx, ty+1) and (tx, ty-1) - Moving vertically.
        # Wait, the cost is "Each time he enters a tile, he pays a toll of 1."
        # Starting tile is free.
        # Moving from (tx, ty) to (tx', ty') costs 1 if it's a different tile.
        # But he can move n units. 
        # If he moves horizontally in the same row y:
        # He is in tile (tx, ty). He moves to (tx', ty).
        # The number of tiles entered is |tx - tx'|.
        # If he moves vertically:
        # He is in tile (tx, ty). He moves to (tx, ty+n).
        # He enters n new tiles.
        
        # This is simply the Manhattan distance in the (tx, ty) space!
        # Distance = |tx_S - tx_T| + |ty_S - ty_T|
    """
    line1 = sys.stdin.readline().split()
    if not line1: return
    sx, sy = map(int, line1)
    line2 = sys.stdin.readline().split()
    if not line2: return
    tx, ty = map(int, line2)

    def get_tile_coords(x, y):
        ty_coord = y
        if y % 2 == 0:
            tx_coord = x // 2
        else:
            tx_coord = (x + 1) // 2
        return tx_coord, ty_coord

    sx_tile, sy_tile = get_tile_coords(sx, sy)
    tx_tile, ty_tile = get_tile_coords(tx, ty)

    print(abs(sx_tile - tx_tile) + abs(sy_tile - ty_tile))

if __name__ == '__main__':
    solve()