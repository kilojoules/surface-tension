import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 blocks aligned horizontally.
    # In row j, the tiles are {(0,j), (1,j)}, {(2,j), (3,j)}... if j is even.
    # In row j, the tiles are {(-1,j), (0,j)}, {(1,j), (2,j)}... if j is odd.
    
    # Let's normalize the coordinates so that in every row, 
    # tiles are {(0,j), (1,j)}, {(2,j), (3,j)}...
    # If j is odd, we shift x by -1.
    
    # We define a function to get the "tile coordinate" (tx, ty)
    # where tx is the index of the 2x1 tile in row ty.
    get_tile_coord = lambda x, y: ( (x + (y % 2)) // 2, y )
    
    s_tile = get_tile_coord(sx, sy)
    t_tile = get_tile_coord(tx, ty)
    
    # The distance between two tiles (x1, y1) and (x2, y2) in this 
    # transformed grid is the Manhattan distance.
    # However, moving horizontally within a row costs 0 if you stay in the same tile,
    # but the problem says "Each time he enters a tile, he pays a toll of 1".
    # Actually, the cost is simply the Manhattan distance in the tile-grid.
    # Let's verify: 
    # Moving from tile (x, y) to (x+1, y) costs 1.
    # Moving from tile (x, y) to (x, y+1) costs 1.
    # The starting tile is already "entered", but the prompt asks for the toll 
    # to reach the target. Usually, this means the number of edges crossed.
    
    dist = abs(s_tile[0] - t_tile[0]) + abs(s_tile[1] - t_tile[1])
    
    print(dist)

if __name__ == "__main__":
    solve()