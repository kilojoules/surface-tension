The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing expressions over statements. For this specific problem, since the optimal path is determined by a mathematical formula based on the coordinates, I will use a direct calculation within a `print` statement, utilizing `abs` for distances and integer division for the tiling logic.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[:2])
    tx, ty = map(int, input_data[2:])

    # The tiling rule: A_{i,j} and A_{i+1,j} are the same tile if i+j is even.
    # This means for a fixed j, tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    # Effectively, the tile boundary in the x-direction is at x=k where k+j is odd.
    
    # To move from (sx, sy) to (tx, ty), we must cross some number of tile boundaries.
    # Moving vertically (changing y) always enters a new tile because tiles are 2x1.
    # Moving horizontally (changing x) might stay within the same tile.
    
    # The cost is the number of tiles entered. 
    # The starting tile is already "entered", but the problem asks for the toll paid.
    # Usually, this implies the number of transitions between different tiles.
    
    # Let's analyze the cost:
    # 1. Vertical distance: Every change in y enters a new tile.
    #    Cost = abs(sy - ty)
    # 2. Horizontal distance: 
    #    If we are at (x, y), we are in a tile that covers x and x+1 if x+y is even.
    #    The cost to move horizontally depends on whether we cross a boundary.
    #    The boundaries are at x=k where k+y is odd.
    #    The number of boundaries between sx and tx at height y is:
    #    If sx < tx: boundaries are k in (sx, tx] such that k+y is odd.
    #    If sx > tx: boundaries are k in (tx, sx] such that k+y is odd.
    
    # However, we can choose the height y at which we move horizontally.
    # If we move horizontally at height y, the cost is the number of k between 
    # sx and tx such that k+y is odd.
    # For any two x-coordinates, there is always a height y (either sy or ty) 
    # that minimizes the number of boundaries crossed.
    # Specifically, the number of boundaries crossed is either floor(dist/2) or ceil(dist/2).
    
    # The total cost is:
    # cost = abs(sy - ty) + (number of x-boundaries crossed)
    # To minimize x-boundaries, we can pick y such that we cross the minimum.
    # The number of boundaries crossed moving from sx to tx at height y is:
    # If (sx + y) is even and (tx + y) is even, we cross (abs(sx-tx)//2) * 2 boundaries? 
    # No. Let's re-evaluate.
    
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are [(0,0),(1,0)], [(2,0),(3,0)], etc.
    # At y=0, boundaries are at x=2, 4, 6...
    # At y=1, boundaries are at x=1, 3, 5...
    
    # Distance dx = abs(sx - tx).
    # At any height y, the number of boundaries crossed is:
    # If (sx+y) is even:
    #   The tiles are [sx, sx+1], [sx+2, sx+3]...
    #   The number of boundaries is dx // 2.
    # If (sx+y) is odd:
    #   The tiles are [sx-1, sx], [sx+1, sx+2]...
    #   The number of boundaries is (dx + 1) // 2.
    
    # We can choose to move horizontally at y=sy or y=ty.
    # The cost is abs(sy - ty) + min(boundaries at sy, boundaries at ty).
    # Note: if sy == ty, we only have one choice of y.
    
    # Let dx = abs(sx - tx)
    # Cost at y: 
    # if (sx + y) % 2 == 0: cost = dx // 2
    # else: cost = (dx + 1) // 2
    
    # We want to minimize this over y in {sy, ty}.
    # But we can also move to any y in between. 
    # Actually, the parity of y only has two states.
    # So we check y = sy and y = sy + 1 (if we can move vertically).
    
    # If sy == ty:
    #   toll = (dx // 2) if (sx + sy) % 2 == 0 else (dx + 1) // 2
    # If sy != ty:
    #   We can pick y to be either parity. The minimum of (dx // 2) and (dx + 1) // 2 
    #   is always dx // 2.
    #   toll = abs(sy - ty) + dx // 2
    
    # Wait, if sy != ty, we must pay for the vertical movement.
    # The vertical movement itself enters new tiles.
    # Let's use the logic:
    # Total Toll = abs(sy - ty) + (dx // 2 if (sy != ty or (sx + sy) % 2 == 0) else (dx + 1) // 2)
    
    # Let's double check Sample 1: 5 0 to 2 5
    # sx=5, sy=0, tx=2, ty=5
    # dx = 3, dy = 5
    # sy != ty, so toll = 5 + (3 // 2) = 5 + 1 = 6? 
    # Sample 1 output is 5. Let's re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Move left 1: (5,0) -> (4,0). A_{5,0} and A_{4,0} are same tile if 4+0 is even. 
    # 4 is even, so they are the same tile. Toll 0.
    # Move up 1: (4,0) -> (4,1). New tile. Toll 1.
    # Move left 1: (4,1) -> (3,1). A_{4,1} and A_{3,1} are same tile if 3+1 is even.
    # 4 is even, so they are the same tile. Toll 0.
    # Move up 3: (3,1) -> (3,4). 3 new tiles. Toll 3.
    # Move left 1: (3,4) -> (2,4). A_{3,4} and A_{2,4} are same tile if 2+4 is even.
    # 6 is even, so they are the same tile. Toll 0.
    # Move up 1: (2,4) -> (2,5). New tile. Toll 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # In this path, he moved horizontally at y=0, y=1, and y=4.
    # At y=0, x: 5->4 (same tile because 4+0 even)
    # At y=1, x: 4->3 (same tile because 3+1 even)
    # At y=4, x: 3->2 (same tile because 2+4 even)
    # He crossed dx=3 horizontal units using 0 toll by picking y such that (x+y) was even.
    # He crossed dy=5 vertical units. Each vertical move enters a new tile.
    # Total toll = 5 + 0 = 5.
    
    # Is it always possible to cross dx with 0 toll if we can change y?
    # To move from x to x-1 without toll, we need (x-1)+y to be even.
    # This means y must have the same parity as x-1.
    # If we can pick y to be any integer between sy and ty, we can 
    # potentially cross all dx units with 0 toll if we can switch y 
    # for each unit. But switching y costs 1.
    # Actually, the rule is: we can move any n units in one direction.
    # If we move horizontally at height y, we cross a boundary every 2 units.
    # The number of boundaries is dx // 2 if we start at the "right" side of the tile.
    
    # Let's re-evaluate:
    # To move from sx to tx, we must cross some boundaries.
    # A boundary exists at x=k if k+y is odd.
    # For a fixed y, the number of boundaries between sx and tx is:
    # count(k in (min(sx,tx), max(sx,tx)) such that k+y is odd) + 1 (if the end is a boundary)
    # Actually, the number of tiles is:
    # If sx == tx: cost is abs(sy - ty)
    # If sx != tx:
    # We can move to a height y, then move horizontally, then to ty.
    # The cost is abs(sy - y) + (number of tiles crossed horizontally) + abs(y - ty).
    # The number of tiles crossed horizontally is:
    # If we are at (x, y), we are in tile T. The tile covers {x, x+1} if x+y is even.
    # To move from sx to tx, the number of tiles is:
    # If (sx+y) is even, we are in tile {sx, sx+1}.
    # The boundaries are at y+k = odd.
    # The number of tiles is (abs(sx-tx) + 1) // 2 if (sx+y) is odd, 
    # and abs(sx-tx) // 2 if (sx+y) is even.
    
    # Total cost = abs(sy - y) + abs(y - ty) + (abs(sx-tx) // 2 if (sx+y)%2==0 else (abs(sx-tx)+1)//2)
    # We want to minimize this over y.
    # Note that abs(sy - y) + abs(y - ty) is minimized when y is between sy and ty, 
    # and its value is abs(sy - ty).
    # If sy != ty, we can pick y to be either parity, so we can always achieve the (dx // 2) cost.
    # If sy == ty, we are stuck with the parity of sy.
    
    # Final Logic:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # if sy != ty:
    #     ans = dy + dx // 2
    # else:
    #     ans = (dx // 2) if (sx + sy) % 2 == 0 else (dx + 1) // 2
    # Wait, let's check Sample 2: 3 1 to 4 1
    # sx=3, sy=1, tx=4, ty=1
    # dx=1, dy=0. sy==ty.
    # (sx+sy)%2 = (3+1)%2 = 0.
    # ans = 1 // 2 = 0. Correct.
    
    # Sample 1: 5 0 to 2 5
    # dx=3, dy=5. sy!=ty.
    # ans = 5 + 3 // 2 = 5 + 1 = 6. 
    # Still getting 6, but sample says 5. Let me re-read again.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # In Sample 1:
    # (5, 0) -> (4, 0): Same tile (4+0 is even). Toll 0.
    # (4, 0) -> (4, 1): New tile. Toll 1.
    # (4, 1) -> (3, 1): Same tile (3+1 is even). Toll 0.
    # (3, 1) -> (3, 4): New tiles. Toll 3.
    # (3, 