The constraint to avoid explicit loops and recursion forces a functional programming style, encouraging the use of declarative expressions over imperative state changes. For this problem, I will use `abs()` for distance calculations and `max()`/`min()` to determine the cost based on the tile layout logic, utilizing a single `print()` statement to output the result.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tile layout is based on the parity of i + j.
    # When i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 blocks aligned horizontally.
    # The pattern shifts every row.
    # In row y, if y is even: tiles are [0,1], [2,3], ... (i is even)
    # In row y, if y is odd: tiles are [-1,0], [1,2], ... (i is odd)
    # Essentially, in row y, a tile boundary exists between x and x+1 if (x + y) is odd.
    
    # To move from (sx, sy) to (tx, ty):
    # 1. Vertical movement: Every change in y enters a new tile.
    #    Cost = |sy - ty|
    # 2. Horizontal movement: 
    #    Within a row y, we move from sx to tx.
    #    A toll is paid if we cross a boundary (x, x+1) where (x + y) is odd.
    #    However, the problem says "Each time he enters a tile, he pays a toll".
    #    Starting tile is free.
    #    If we move horizontally in row sy, and then vertically, 
    #    we can optimize which row we use for horizontal travel.
    
    # Let's analyze the cost:
    # The cost to move between (sx, sy) and (tx, ty) is:
    # dy = |sy - ty|
    # dx = |sx - tx|
    # In any row y, the number of tiles crossed to get from sx to tx is:
    # If we are at (sx, sy) and move to (tx, sy), the number of boundaries crossed is
    # the number of x in [min(sx, tx), max(sx, tx)-1] such that (x + sy) is odd.
    
    # Let f(x1, x2, y) be the number of boundaries crossed.
    # The boundaries are at x + 0.5 where x + y is odd.
    # This is equivalent to counting x in range(min(sx, tx), max(sx, tx)) where (x+y)%2 == 1.
    
    # The total cost is dy + (cost of horizontal movement in some row y).
    # We can pick any row y between sy and ty (inclusive) to perform the horizontal move.
    # Or we can move horizontally in row sy or row ty.
    # The number of boundaries crossed in row y is:
    # Let L = min(sx, tx), R = max(sx, tx).
    # Count x in {L, ..., R-1} such that x % 2 != y % 2.
    # This count is (R - L) // 2 if (R - L) is even.
    # If (R - L) is odd, it's either (R - L) // 2 or (R - L) // 2 + 1 depending on y.
    
    # Specifically, the number of boundaries is:
    # floor((R - L + 1) / 2) if we pick the "better" row.
    # But we can only pick row y if we actually visit it.
    # Since we must travel from sy to ty, we visit all rows between them.
    # If sy == ty, we are stuck in one row.
    # If sy != ty, we can pick y = sy or y = ty or any in between.
    
    # Let's refine:
    # Cost = |sy - ty| + (horizontal cost)
    # Horizontal cost in row y:
    # The boundaries are at x + 0.5 where x+y is odd.
    # For a fixed y, the number of boundaries between sx and tx is:
    # Let L = min(sx, tx), R = max(sx, tx).
    # The boundaries are x = L, L+1, ..., R-1.
    # Boundary x is crossed if (x + y) % 2 == 1.
    # In any two adjacent x's, exactly one is odd.
    # So in R-L boundaries, there are (R-L)//2 boundaries of one parity and (R-L+1)//2 of the other.
    # We can choose y to minimize this, provided we can reach that row.
    # If sy != ty, we can always pick y such that we cross only (R-L)//2 boundaries.
    # If sy == ty, we are forced to use y = sy.
    
    # Wait, the rule is: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    # This means the boundary between A_{i,j} and A_{i+1,j} is REMOVED if i+j is even.
    # A boundary exists between x and x+1 in row y if (x + y) is ODD.
    # This matches my logic.
    
    # Let's calculate:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # If sy == ty:
    #   Horizontal cost is the number of x in [min(sx, tx), max(sx, tx)-1] such that (x + sy) % 2 == 1.
    #   This is (dx + 1) // 2 if (sx + sy) % 2 == 0 and dx % 2 == 1, else dx // 2.
    #   Actually, it's simpler: 
    #   The boundaries are at x+0.5 where x+y is odd.
    #   If sx < tx, boundaries are x = sx, sx+1, ..., tx-1.
    #   Number of odd (x+sy) is:
    #   (tx - sx + 1) // 2 if (sx + sy) % 2 == 1 or (tx + sy) % 2 == 1 else (tx - sx) // 2.
    #   Correct logic for count of x in [L, R-1] such that (x+y)%2 == 1:
    #   Total elements = R - L.
    #   If R-L is even, count is (R-L)//2.
    #   If R-L is odd, count is (R-L)//2 + 1 if (L+y)%2 == 1, else (R-L)//2.
    
    # If sy != ty:
    #   We can pick y = sy or y = ty.
    #   One of them will give (R-L)//2, the other might give (R-L+1)//2.
    #   So we can always achieve (R-L)//2.
    
    # Final formula:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # if sy == ty:
    #     res = dy + ((dx + 1) // 2 if (min(sx, tx) + sy) % 2 == 1 and dx % 2 == 1 else dx // 2)
    #     # Wait, if (L+y) is odd, the first boundary is crossed. 
    #     # If L=0, R=1, y=0: x=0 is even. (0+0)%2=0. Cost 0.
    #     # If L=0, R=1, y=1: x=0 is odd. (0+1)%2=1. Cost 1.
    #     # The number of x in [L, R-1] such that (x+y)%2 == 1 is:
    #     # (R-L)//2 + (1 if (R-L)%2 == 1 and (L+y)%2 == 1 else 0)
    # else:
    #     res = dy + dx // 2
    
    # Let's double check Sample 1: 5 0, 2 5
    # sx=5, sy=0, tx=2, ty=5
    # dx = 3, dy = 5
    # sy != ty, so res = 5 + 3 // 2 = 5 + 1 = 6? 
    # Sample 1 output is 5. Let me re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Move left 1: (5,0) -> (4,0). A_{5,0} and A_{4,0} are same tile if 4+0 is even. 
    # 4 is even, so they are the same tile. Cost 0.
    # Move up 1: (4,0) -> (4,1). Always enters a new tile. Cost 1.
    # Move left 1: (4,1) -> (3,1). A_{4,1} and A_{3,1} are same tile if 3+1 is even.
    # 4 is even, so they are the same tile. Cost 0.
    # Move up 3: (3,1) -> (3,4). Cost 3.
    # Move left 1: (3,4) -> (2,4). A_{3,4} and A_{2,4} are same tile if 2+4 is even.
    # 6 is even, so they are the same tile. Cost 0.
    # Move up 1: (3,4) -> (3,5). Cost 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # My manual trace:
    # (5,0) -> (4,0) : Tile(4,0 & 5,0) -> Tile(4,0 & 5,0). Cost 0.
    # (4,0) -> (4,1) : Tile(4,0 & 5,0) -> Tile(4,1 & 3,1). Cost 1.
    # (4,1) -> (3,1) : Tile(4,1 & 3,1) -> Tile(4,1 & 3,1). Cost 0.
    # (3,1) -> (3,4) : Tile(4,1 & 3,1) -> Tile(3,2 & 2,2) -> Tile(3,3 & 4,3) -> Tile(3,4 & 2,4). Cost 3.
    # (3,4) -> (2,4) : Tile(3,4 & 2,4) -> Tile(3,4 & 2,4). Cost 0.
    # (2,4) -> (2,5) : Tile(3,4 & 2,4) -> Tile(2,5 & 1,5). Cost 1.
    # Total = 5.
    
    # The cost is simply dy + (dx // 2) if we can pick the parity of the row.
    # But wait, in Sample 1: dx=3, dy=5. 5 + 3//2 = 6. Still not 5.
    # Let's re-evaluate.
    # In the trace, he moved horizontally at y=0, y=1, and y=4.
    # At y=0, he moved 5 -> 4. (4+0) is even, so A_{4,0} and A_{5,0} are one tile.
    # At y=1, he moved 4 -> 3. (3+1) is even, so A_{3,1} and A_{4,1} are one tile.
    # At y=4, he moved 3 -> 2. (2+4) is even, so A_{2,4} and A_{3,4} are one tile.
    # He covered dx=3 using 3 different rows, and in each row the move was "free".
    # This means if dy >= dx, the horizontal cost can be 0!
    # Because he can alternate rows to make every horizontal step free.
    # If dy < dx, he can make dy steps free, and the remaining (dx - dy) must be paid.
    # The remaining horizontal distance is (dx - dy). 
    # In a single row, the cost is (dist + 1) // 2.
    # So cost = dy + (dx - dy + 1) // 