import sys

def solve():
    # Read input and parse integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The problem describes a grid of 2x1 tiles.
    # A tile covers A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j, the tiles are [(0,1), (2,3), ...] if j is even
    # and [(-1,0), (1,2), ...] if j is0 is odd.
    # Essentially, a tile is defined by the pair (j, floor((i + (j % 2)) / 2)).
    # Let's transform the coordinates to a new coordinate system (u, v)
    # where u is the tile index horizontally and v is the tile index vertically.
    # For a cell (i, j):
    # v = j
    # u = (i + (j % 2)) // 2
    
    # However, the movement rules are: moving n units in a direction.
    # Moving vertically (changing j) always enters a new tile unless n=0.
    # Moving horizontally (changing i) might stay in the same tile.
    # A move is "free" if it stays within the same tile.
    # The cost is the number of tiles entered.
    # This is equivalent to the distance in a graph where nodes are tiles.
    # Two tiles are connected if they share a boundary.
    # The distance between tile (u1, v1) and (u2, v2) in this specific 
    # brick-wall pattern is known to be:
    # dist = max(|u1 - u2|, |v1 - v2| + (1 if parity of u/v changes in a specific way else 0))
    # Actually, the simplest formula for this specific grid distance is:
    # Let dx = tx - sx, dy = ty - sy.
    # The cost is max(abs(dx // 2), abs(dy)) if we align the bricks.
    # A more robust way:
    # The distance is max(|sx - tx| // 2, |sy - ty|) 
    # but we must account for the offset shift every row.
    # The correct formula for this specific tiling distance is:
    # dist = max(abs(sx - tx + (sy % 2) - (ty % 2)) // 2, abs(sy - ty))
    # Wait, the standard formula for this "brick" distance is:
    # Let x' = x + (y % 2) / 2
    # Distance = max(|x1' - x2'|, |y1 - y2|)
    # Since we are at centers (.5, .5), we can use:
    # cost = max(abs(sx + (sy % 2) - (tx + (ty % 2))) // 2, abs(sy - ty))
    # Let's test Sample 1: 5 0 -> 2 5
    # sx=5, sy=0; tx=2, ty=5
    # abs(5 + 0 - (2 + 1)) // 2 = 2 // 2 = 1
    # abs(0 - 5) = 5
    # max(1, 5) = 5. Correct.
    # Sample 2: 3 1 -> 4 1
    # sx=3, sy=1; tx=4, ty=1
    # abs(3 + 1 - (4 + 1)) // 2 = abs(4 - 5) // 2 = 0
    # abs(1 - 1) = 0
    # max(0, 0) = 0. Correct.
    
    # The logic: 
    # In one vertical step, you move 1 unit of distance.
    # In two horizontal steps, you move 1 unit of "tile" distance.
    # Because the bricks shift, the horizontal distance is (dx + shift) // 2.
    
    res = max(abs(sx + (sy % 2) - (tx + (ty % 2))) // 2, abs(sy - ty))
    print(res)

if __name__ == "__main__":
    solve()