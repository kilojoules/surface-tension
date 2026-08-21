import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3)... if j is even
    # and (-1,0), (1,2)... if j is odd.
    # Essentially, a tile is defined by (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a space where each tile is a 1x1 unit.
    # The y-coordinate remains the same because tiles are always 1 unit high.
    # The x-coordinate of the tile is (i + (j % 2)) // 2.
    
    # Starting tile coordinates:
    s_tile_x = (sx + (sy % 2)) // 2
    s_tile_y = sy
    
    # Target tile coordinates:
    t_tile_x = (tx + (ty % 2)) // 2
    t_tile_y = ty
    
    # The distance in a grid where you can move any distance in one of 4 directions
    # and pay 1 per tile entered is the L1 distance (Manhattan distance),
    # but since we start inside a tile, the first tile is free.
    # The cost to move from (sx, sy) to (tx, ty) in this transformed grid is:
    # cost = abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y)
    
    # However, the problem says "Each time he enters a tile, he pays a toll of 1."
    # If he is already in a tile, moving within it costs 0.
    # Moving to a different tile costs 1.
    # The minimum number of tile boundaries crossed is the Manhattan distance 
    # between the tile indices.
    
    print(abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y))

if __name__ == "__main__":
    solve()