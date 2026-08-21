import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiles are 2x1 blocks. 
    # According to the rule: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    # Essentially, a tile is defined by the pair (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a coordinate system where each unit 
    # represents a tile boundary.
    # For a given y, the tile index in x is x_tile = (x + (y % 2)) // 2.
    # The y index is simply y.
    
    # Starting tile coordinates:
    s_tile_x = (sx + (sy % 2)) // 2
    s_tile_y = sy
    
    # Target tile coordinates:
    t_tile_x = (tx + (ty % 2)) // 2
    t_tile_y = ty
    
    # The distance in a grid where you can move any distance in one of 4 directions
    # is the L1 distance (Manhattan distance), but the cost is based on how many
    # tiles you enter. 
    # Moving from (sx, sy) to (tx, ty) in the tile-grid:
    # The cost is |s_tile_x - t_tile_x| + |s_tile_y - t_tile_y|.
    # However, we must account for the fact that moving diagonally in the 
    # original coordinate system might be cheaper.
    # Actually, the problem is equivalent to finding the distance in a graph
    # where nodes are tiles.
    # From tile (tx, ty), you can reach:
    # - (tx +/- 1, ty) with cost 1
    # - (tx, ty +/- 1) with cost 1
    # But wait, the "move n units" rule means you can jump over tiles.
    # If you move vertically, you enter every tile in between.
    # If you move horizontally, you might stay in the same tile or enter a new one.
    
    # Let's re-evaluate:
    # To get from (s_tile_x, s_tile_y) to (t_tile_x, t_tile_y):
    # The cost is max(|s_tile_x - t_tile_x|, |s_tile_y - t_tile_y|) 
    # is NOT correct because we can only move in 4 cardinal directions.
    # The cost is simply the Manhattan distance in the tile-coordinate system:
    # cost = |s_tile_x - t_tile_x| + |s_tile_y - t_tile_y|
    # But there is a catch: moving diagonally in the tile grid (changing both x and y)
    # can be done by moving 1 unit horizontally and 1 unit vertically.
    # The actual minimum cost to move between tiles in this specific tiling 
    # is known to be:
    # dist = max(|s_tile_x - t_tile_x|, |s_tile_y - t_tile_y|, (|s_tile_x - t_tile_x| + |s_tile_y - t_tile_y| + 1) // 2)
    # Wait, the simplest derivation for this specific tiling problem is:
    # The distance is max(|s_tile_x - t_tile_x|, |s_tile_y - t_tile_y|, 
    #                     (|s_tile_x - t_tile_x| + |s_tile_y - t_tile_y| + 1) // 2)
    # Actually, the correct formula for this specific grid is:
    # cost = max(abs(s_tile_x - t_tile_x), abs(s_tile_y - t_tile_y), 
    #             (abs(s_tile_x - t_tile_x) + abs(s_tile_y - t_tile_y) + 1) // 2)
    # Let's test Sample 1: S(5,0), T(2,5)
    # s_tile_x = (5 + 0)//2 = 2, s_tile_y = 0
    # t_tile_x = (2 + (5%2))//2 = (2+1)//2 = 1, t_tile_y = 5
    # dx = |2-1| = 1, dy = |0-5| = 5
    # max(1, 5, (1+5+1)//2) = max(1, 5, 3) = 5. Correct.
    
    # Sample 2: S(3,1), T(4,1)
    # s_tile_x = (3 + 1)//2 = 2, s_tile_y = 1
    # t_tile_x = (4 + 1)//2 = 2, t_tile_y = 1
    # dx = 0, dy = 0 -> 0. Correct.
    
    dx = abs(s_tile_x - t_tile_x)
    dy = abs(s_tile_y - t_tile_y)
    
    print(max(dx, dy, (dx + dy + 1) // 2))

if __name__ == "__main__":
    solve()