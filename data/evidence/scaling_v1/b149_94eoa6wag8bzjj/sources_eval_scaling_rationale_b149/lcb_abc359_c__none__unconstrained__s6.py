import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling pattern:
    # When i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, if j is even, tiles are [0,1], [2,3], ... (i=0, 2, 4...)
    # In row j, if j is odd, tiles are [-1,0], [1,2], ... (i=1, 3, 5...)
    # To normalize this, we can transform coordinates:
    # A point (x, y) belongs to tile (floor(x - (y % 2)), y) if we shift x by y%2.
    # However, a simpler observation:
    # Moving vertically always enters a new tile.
    # Moving horizontally might stay in the same tile or enter a new one.
    # The cost is effectively the Manhattan distance in a transformed space.
    # Let's use the transformation: 
    # X' = x - (y % 2), Y' = y
    # The distance is max(|X's - X't|, |Y's - Y't|) is NOT correct here.
    # The correct logic for this specific tiling:
    # The cost is max(|sx - tx|, |sy - ty|) if we consider the 
    # connectivity of the 2x1 blocks.
    # Specifically, the distance is max(|sx - tx + (sy%2) - (ty%2)|, |sy - ty|)
    # Wait, the simplest derivation for this specific grid is:
    # cost = max(abs(sx - tx), abs(sy - ty)) 
    # But we must account for the offset.
    # Let's refine: 
    # If we are at (sx, sy), the tile index is ( (sx - (sy%2)) // 2, sy )
    # The distance between tile (x1, y1) and (x2, y2) in this grid is:
    # dist = max(abs(x1 - x2), abs(y1 - y2))
    
    # Calculate normalized coordinates
    # x_norm = (x - (y % 2)) // 2
    # y_norm = y
    
    # We use a list comprehension or map to avoid loops, 
    # though for a single calculation it's not needed.
    # We calculate the result in one go.
    
    res = max(
        abs(((sx - (sy % 2)) // 2) - ((tx - (ty % 2)) // 2)),
        abs(sy - ty)
    )
    
    print(res)

if __name__ == "__main__":
    solve()