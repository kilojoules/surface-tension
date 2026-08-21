import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is based on the parity of i + j.
    # When i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This creates a brick-like pattern.
    # Let's transform the coordinates to a system where we can calculate 
    # the cost based on the number of tile boundaries crossed.
    # In this specific tiling, moving vertically always crosses a tile boundary
    # unless you are already in the target tile.
    # Moving horizontally might not cross a boundary if you stay within a 2x1 tile.
    
    # The cost to move between (sx, sy) and (tx, ty) in this grid is:
    # cost = max(|sx - tx|, |sy - tx|) is NOT correct here.
    # The correct logic for this specific tiling:
    # Each vertical step costs 1.
    # Horizontal steps cost 0 if they stay within the 2x1 block.
    # The optimal strategy is to move diagonally in a sense.
    # The distance is max(|sx - tx|, |sy - ty|) if we consider the 
    # connectivity of the tiles.
    # Specifically, for this tiling, the distance is:
    # Let dx = |sx - tx| and dy = |sy - ty|
    # The cost is max(dx // 2, dy) if we align correctly, 
    # but the parity of the coordinates matters.
    
    # A more robust observation for this specific problem:
    # The cost is max(abs(sx - tx) // 2, abs(sy - ty)) 
    # is almost correct, but we must account for the offset.
    # The tiles are 2x1. In every 2 units of x, we must have moved 
    # at least 1 unit of y to 'jump' between the staggered bricks.
    # The actual minimum cost is max(abs(sx - tx + 1) // 2, abs(sy - ty)) 
    # is also not quite it.
    
    # Correct logic:
    # To move from (sx, sy) to (tx, ty):
    # 1. Vertical distance dy = abs(sy - ty) always costs dy.
    # 2. Horizontal distance dx = abs(sx - tx).
    # Because tiles are 2x1 and staggered, you can cover 2 units of x 
    # for every 1 unit of y.
    # The cost is max(dy, (dx + 1) // 2) if we are lucky with parity,
    # but since we can move n units in one direction, we can 
    # effectively move diagonally.
    # The cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # is only true if the start and end are aligned.
    
    # Let's use the coordinate transformation:
    # A point (x, y) is in a tile. 
    # If (x + y) is even, tile is {(x, y), (x+1, y)}
    # If (x + y) is odd, tile is {(x-1, y), (x, y)}
    # This is equivalent to saying tile ID is (x // 2, y) if y is even
    # and ((x+1) // 2, y) if y is odd.
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's redefine: 
    # If j is even: tiles are {(0,0),(1,0)}, {(2,0),(3,0)}... -> x' = x // 2
    # If j is odd: tiles are {(-1,1),(0,1)}, {(1,1),(2,1)}... -> x' = (x+1) // 2
    # This is exactly what happens if we transform x based on the parity of y.
    
    # Let's use a simpler observation:
    # The distance is max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # is almost correct, but the parity of sx+sy and tx+ty matters.
    # The most reliable formula for this specific problem is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # however, if we move purely horizontally, we might need an extratoll.
    # Actually, the simplest form is:
    # ans = max(abs(sy - ty), (abs(sx - tx) + 1) // 2)
    # Let's check Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+1)//2) = 1. 
    # Wait, Sample 2 output is 0. 
    # In Sample 2: sx=3, sy=1. i+j = 3+1 = 4 (even). 
    # So A_{3,1} and A_{4,1} are in the same tile. 
    # Thus moving from 3.5, 1.5 to 4.5, 1.5 costs 0.
    
    # The rule: A_{i,j} and A_{i+1,j} are same tile if i+j is even.
    # This means for a fixed j, the tiles are:
    # If j is even: [0,1], [2,3], [4,5]... (i is even)
    # If j is odd: [-1,0], [1,2], [3,4]... (i is odd)
    # In both cases, the tile index is (i + (j % 2)) // 2.
    
    # Let X(i, j) = (i + (j % 2)) // 2
    # Let Y(i, j) = j
    # The distance is then the Manhattan distance in the transformed grid?
    # No, because we can move any n units.
    # The cost is max(|X(sx, sy) - X(tx, ty)|, |Y(sx, sy) - Y(tx, ty)|)
    # Let's check Sample 2: X(3, 1) = (3 + 1)//2 = 2. X(4, 1) = (4 + 1)//2 = 2.
    # Y(3, 1) = 1, Y(4, 1) = 1. max(|2-2|, |1-1|) = 0. Correct.
    # Sample 1: X(5, 0) = (5+0)//2 = 2. X(2, 5) = (2+1)//2 = 1.
    # Y(5, 0) = 0, Y(2, 5) = 5. max(|2-1|, |0-5|) = 5. Correct.
    
    # Final formula: max(abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2), abs(sy - ty))
    
    res = max(abs((sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2), abs(sy - ty))
    print(res)

if __name__ == "__main__":
    solve()