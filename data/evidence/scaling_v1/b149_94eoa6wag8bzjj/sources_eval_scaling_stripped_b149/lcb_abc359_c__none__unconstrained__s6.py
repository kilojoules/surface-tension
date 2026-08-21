import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiles are 2x1. 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3)... if j is even
    # and pairs (1,2), (3,4)... if j is odd.
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # For a cell (x, y), it belongs to tile:
    # If (x + y) is even, the tile is {(x, y), (x+1, y)}. 
    # We can represent this tile by the coordinate (x // 2, y) if we shift x based on y.
    # Specifically, if y is even, tiles are [0,1], [2,3]... -> x // 2
    # If y is odd, tiles are [-1,0], [1,2]... -> (x-1) // 2
    
    # A simpler way to think about the tile ID for cell (x, y):
    # The tile is horizontal. Its y-coordinate is simply y.
    # Its x-coordinate depends on whether y is even or odd.
    # If y is even, tile index is x // 2 (for x=0,1 -> 0; x=2,3 -> 1)
    # If y is odd, tile index is (x-1) // 2 (for x=1,2 -> 0; x=3,4 -> 1)
    # Note: for y odd, x=0 belongs to the tile ending at 0, so (0-1)//2 = -1.
    
    def get_tile_coords(x, y):
        # If y is even, tiles are {(0,0),(1,0)}, {(2,0),(3,0)}...
        # If y is odd, tiles are {(-1,1),(0,1)}, {(1,1),(2,1)}...
        # The tile index is (x - (y % 2)) // 2
        return (x - (y % 2)) // 2, y

    s_tile_x, s_tile_y = get_tile_coords(sx, sy)
    t_tile_x, t_tile_y = get_tile_coords(tx, ty)

    # The distance between two tiles (x1, y1) and (x2, y2) in a grid 
    # where you can moveに 1 unit (cost 1) is the Manhattan distance.
    # However, the problem says "Each time he enters a tile, he pays a toll of 1."
    # This is equivalent to the distance between the starting tile and target tile.
    # Distance = |x1 - x2| + |y1 - y2|
    
    ans = abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y)
    print(ans)

if __name__ == "__main__":
    solve()