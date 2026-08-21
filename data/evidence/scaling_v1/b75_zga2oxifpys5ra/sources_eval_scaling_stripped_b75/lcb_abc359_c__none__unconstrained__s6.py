import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling pattern is a brick-like layout.
    # A tile covers A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j, the tiles are {(0,j), (1,j)}, {(2,j), (3,j)}... if j is even
    # and {(-1,j), (0,j)}, {(1,j), (2,j)}... if j is odd.
    
    # Let's normalize the coordinates to a grid where each unit is a tile.
    # In row j, the tile index is floor((i + (j % 2)) / 2).
    # Let X(i, j) = (i + (j % 2)) // 2
    # Let Y(i, j) = j
    
    # The distance between two tiles (X1, Y1) and (X2, Y2) in this specific 
    # grid movement (where moving vertically costs 1 and moving horizontally 
    # within the same 'brick' costs 0) is equivalent to the Manhattan distance
    # on the transformed grid, but with a specific cost structure.
    
    # Specifically, moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1, UNLESS the move is 
    # covered by the 2x1 tile.
    
    # The distance is max(|X1 - X2|, |Y1 - Y2|) if we could move diagonally.
    # But we can only move U, D, L, R.
    # The cost is actually:
    # cost = abs(Y1 - Y2) + max(0, abs(X1 - X2) - abs(Y1 - Y2)//2 - (1 if abs(Y1-Y2)%2 != 0 else 0))
    # Wait, a simpler observation:
    # The distance is simply the Manhattan distance on the transformed grid 
    # divided by 2, but handled carefully.
    
    # Let's use the property:
    # The distance is max(|X1 - X2|, (|X1 - X2| + |Y1 - Y2| + 1) // 2) 
    # is not correct.
    
    # Correct logic for this specific tiling:
    # The distance is simply:
    # dx = abs(X1 - X2)
    # dy = abs(Y1 - Y2)
    # result = max(dx, (dx + dy + 1) // 2) 
    # Actually, the most reliable formula for this problem is:
    # dist = max(abs(X1 - X2), (abs(X1 - X2) + abs(Y1 - Y2) + 1) // 2)
    # Let's verify with Sample 1: S(5,0), T(2,5)
    # X1 = (5 + 0)//2 = 2, Y1 = 0
    # X2 = (2 + (5%2))//2 = (2+1)//2 = 1, Y2 = 5
    # dx = |2-1| = 1, dy = |0-5| = 5
    # max(1, (1+5+1)//2) = max(1, 3) = 3. (Incorrect, Sample 1 says 5)
    
    # Re-evaluating:
    # The cost to move from (X1, Y1) to (X2, Y2) is:
    # If we move vertically, we always enter a new tile.
    # If we move horizontally, we enter a new tile every 2 units of x (usually).
    # The distance is actually:
    # cost = abs(Y1 - Y2) + max(0, abs(X1 - X2) - (abs(Y1 - Y2) + 1) // 2)
    # Wait, let's try Sample 1 again: X1=2, Y1=0, X2=1, Y2=5
    # cost = 5 + max(0, 1 - (5+1)//2) = 5 + max(0, 1-3) = 5. (Correct!)
    
    # Sample 2: S(3,1), T(4,1)
    # X1 = (3 + 1)//2 = 2, Y1 = 1
    # X2 = (4 + 1)//2 = 2, Y2 = 1
    # cost = 0 + max(0, 0 - 0) = 0. (Correct!)
    
    # Let's check the logic:
    # To move dy vertically, you must pay dy.
    # While moving vertically, you can shift your X coordinate by 1 every 2 vertical steps
    # for "free" (because the tiles shift).
    # The number of free X-shifts provided by dy vertical steps is (dy + 1) // 2.
    # Any remaining X-distance must be paid for.
    
    x1 = (sx + (sy % 2)) // 2
    y1 = sy
    x2 = (tx + (ty % 2)) // 2
    y2 = ty
    
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    
    # The number of X-steps covered by the "zigzag" of the bricks is (dy + 1) // 2
    # However, the parity of the start and end Y matters.
    # If we move from y1 to y2, we cross dy boundaries.
    # The number of X-shifts we get for free is dy // 2.
    # If dy is odd, we can get one more shift depending on the direction.
    # Actually, the formula is:
    # cost = dy + max(0, dx - (dy + 1) // 2) if we can optimize.
    # But the "free" shift only happens if we change Y.
    # Let's use: cost = max(dy, dx + (dy // 2)) is also wrong.
    
    # Correct logic:
    # Each vertical step costs 1.
    # Two vertical steps allow us to move 1 unit of X for free.
    # So for dy vertical steps, we get dy // 2 free X-steps.
    # If dy is odd, the last step also allows a free X-shift if the 
    # parity of the current X and Y matches the tile boundary.
    # The total cost is dy + max(0, dx - (dy + 1) // 2) is almost there.
    # Let's refine:
    # The distance is dy + max(0, dx - (dy + 1) // 2) if we start/end at the 
    # "correct" side of the brick.
    # Actually, the simplest form is:
    # result = max(dy, dx + (dy // 2)) if dy % 2 == 0 else max(dy, dx + (dy // 2))
    # No, let's use the logic:
    # To cover dx, we can use dy // 2 free moves.
    # If dy is odd, we can get one more free move if the parity of 
    # (x1, y1) and (x2, y2) allows it.
    # But since we can choose to move 1 unit vertically at the start or end,
    # we can always get (dy + 1) // 2 free moves if we are clever.
    # Wait, the sample 1: dx=1, dy=5. (5+1)//2 = 3. 5 + max(0, 1-3) = 5.
    # If dx=4, dy=5: 5 + max(0, 4-3) = 6.
    
    print(dy + max(0, dx - (dy + 1) // 2))

if __name__ == "__main__":
    solve()