import sys

def solve():
    # Read input from stdin
    # Use map and split to extract coordinates without explicit loops
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiles are 2x1. When i+j is even, (i,j) and (i+1,j) are one tile.
    # This means for a fixed j, the tiles are aligned at x = ... -2, 0, 2 ... if j is even
    # and x = ... -1, 1, 3 ... if j is odd.
    # Let's transform the coordinates to a system where each unit move in Y 
    # costs 1, and moves in X cost based on the tile boundaries.
    
    # A point (x, y) is in tile (floor(x/2), y) if y is even
    # and (floor((x-1)/2), y) if y is odd.
    # However, a simpler way to view this is:
    # The cost to move from (sx, sy) to (tx, ty) is the L1 distance 
    # in a transformed space.
    # Let u = x + y and v = x - y. 
    # The boundaries are diagonals. 
    # Actually, the cost is simply:
    # cost = abs(sy - ty) + abs( (sx + (sy % 2)) // 2 - (tx + (ty % 2)) // 2 )
    # But we must account for the fact that we can move horizontally 
    # within a tile for free.
    
    # Let's define the "tile coordinate" (I, J):
    # J = y
    # I = (x + (y % 2)) // 2
    # The distance between (I1, J1) and (I2, J2) is |I1 - I2| + |J1 - J2|.
    # However, we can move diagonally in the (I, J) space by moving 
    # 1 unit in X and 1 unit in Y in the original space.
    # Specifically, moving from (x, y) to (x+1, y+1):
    # If y is even: (x, y) is in tile ((x+0)//2, y). (x+1, y+1) is in tile ((x+1+1)//2, y+1).
    # This is a move from (I, J) to (I+1, J+1).
    # The cost of moving from (I1, J1) to (I2, J2) is max(|I1 - I2|, |J1 - J2|).
    
    # Let's check Sample 1: (5, 0) to (2, 5)
    # S: I = (5 + 0)//2 = 2, J = 0
    # T: I = (2 + (5%2))//2 = (2+1)//2 = 1, J = 5
    # max(|2-1|, |0-5|) = max(1, 5) = 5. Correct.
    
    # Sample 2: (3, 1) to (4, 1)
    # S: I = (3 + 1)//2 = 2, J = 1
    # T: I = (4 + 1)//2 = 2, J = 1
    # max(|2-2|, |1-1|) = 0. Correct.
    
    # The logic:
    # I_s = (sx + (sy % 2)) // 2
    # J_s = sy
    # I_t = (tx + (ty % 2)) // 2
    # J_t = ty
    # result = max(abs(I_s - I_t), abs(J_s - J_t))
    
    # To avoid multiple lines/assignments and follow the "no loop/recursion" 
    # constraint strictly, I'll use a single print statement with calculations.
    
    print(max(
        abs(((sx + (sy % 2)) // 2) - ((tx + (ty % 2)) // 2)),
        abs(sy - ty)
    ))

if __name__ == "__main__":
    solve()