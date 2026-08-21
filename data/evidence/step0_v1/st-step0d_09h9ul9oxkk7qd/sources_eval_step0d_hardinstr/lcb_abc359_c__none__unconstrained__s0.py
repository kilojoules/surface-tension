The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing data transformation over state updates. For this problem, which requires calculating a distance metric on a grid of $2\times1$ tiles, I will use direct arithmetic expressions to compute the cost based on the parity of coordinates and the Manhattan distance, as the optimal path depends on how many tile boundaries are crossed.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1. 
    # If i+j is even, (i, j) and (i+1, j) are in the same tile.
    # This means for a fixed j, tiles are paired as {(0,j), (1,j)}, {(2,j), (3,j)} if j is even
    # and {(1,j), (2,j)}, {(3,j), (4,j)} if j is odd.
    
    # Let's transform the coordinates to a system where we can calculate the cost.
    # In this grid, moving vertically always crosses a tile boundary.
    # Moving horizontally might not cross a boundary if the two squares are in the same tile.
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # 1. The vertical distance |sy - ty|.
    # 2. The horizontal cost. 
    # If we are at (x, y), the tile boundary is between x and x+1 if (x+y) is odd.
    # The cost to move horizontally is the number of boundaries crossed.
    
    # Let's analyze the horizontal cost at a specific y.
    # A boundary exists between x and x+1 if x+y is odd.
    # The number of boundaries between sx and tx is the number of integers k 
    # between min(sx, tx) and max(sx, tx)-1 such that k+y is odd.
    
    # However, we can change y to minimize the horizontal cost.
    # If we move to a y' where the boundaries are shifted, we might save cost.
    # But we already pay for the vertical movement.
    
    # The optimal strategy is:
    # The cost is max(|sy - ty|, (abs(sx - tx) + abs(sy - ty) + 1) // 2) 
    # is NOT correct here because the tiles are specifically 2x1.
    
    # Correct logic:
    # Each vertical step costs 1.
    # Each horizontal step costs 1, UNLESS we are in a 2x1 tile.
    # In any row y, every second vertical edge is a tile boundary.
    # Specifically, the edge between x and x+1 is a boundary if x+y is odd.
    # To get from sx to tx, we must cross some boundaries.
    # If we stay in row y, we cross ceil(|sx-tx|/2) boundaries if the start/end 
    # positions are favorable, or floor(|sx-tx|/2) if not.
    
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    
    # If we move from (sx, sy) to (tx, ty):
    # We must pay dy for vertical movement.
    # For horizontal, we can pick the best row (either sy or ty or something in between).
    # In any row y, the number of boundaries crossed is:
    # If sx < tx: count k in [sx, tx-1] where k+y is odd.
    # This count is dx // 2 if (sx+y) and (tx+y) have the same parity and 
    # the boundary is not at the ends, etc.
    # Actually, the number of boundaries crossed is:
    # (dx + 1) // 2 if (sx+y) is odd and (tx+y) is even (or vice versa)
    # dx // 2 if (sx+y) is even and (tx+y) is even... 
    # Wait, the simplest way:
    # The number of boundaries crossed is dx // 2 if (sx+y) is even and (tx+y) is even
    # (since the boundary is at k+y=odd, and we start at an even, the first boundary is at sx+1)
    # Let's use the property: boundary at k if k+y is odd.
    # For a fixed y, the number of boundaries between sx and tx is:
    # floor((max(sx, tx) + (y % 2)) / 2) - floor((min(sx, tx) + (y % 2)) / 2)
    # This is essentially (dx + 1) // 2 or dx // 2.
    
    # The total cost is dy + (horizontal cost).
    # We can choose y to be sy or ty.
    # If we use row sy, horizontal cost is h(sx, tx, sy).
    # If we use row ty, horizontal cost is h(sx, tx, ty).
    # But we can also move diagonally.
    # The actual minimum cost is:
    # cost = dy + max(0, (dx - (1 if (sx+sy)%2 == 0 and (tx+sy)%2 == 0 else 0) - (1 if ...)) // 2)
    # Actually, the simplest formula for this specific tile layout is:
    # cost = max(dy, (dx + dy + 1) // 2) 
    # Let's check Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+0+1)//2) = 1? No, Sample 2 is 0.
    # In Sample 2: sx=3, sy=1. sx+sy = 4 (even). A_{3,1} and A_{4,1} are in the same tile.
    # So moving from 3.5, 1.5 to 4.5, 1.5 costs 0.
    
    # Let's re-evaluate:
    # A boundary exists between x and x+1 if x+y is odd.
    # If sx+sy is even, the tile covers [sx, sx+1] and [sx+1, sx+2].
    # So the boundary is at sx+1 if sx+sy is odd.
    # If sx+sy is even, the boundary is at sx-1 and sx+2.
    
    # Let's use the coordinate transformation:
    # A tile is defined by ( (x+y)//2, y ) if x+y is even.
    # This is getting complex. Let's use the property:
    # The cost is dy + (number of boundaries crossed horizontally).
    # We can pick any y in [min(sy, ty), max(sy, ty)].
    # For a fixed y, the number of boundaries between sx and tx is:
    # Let x1 = min(sx, tx), x2 = max(sx, tx).
    # Boundary at k if k+y is odd.
    # Number of k in {x1, ..., x2-1} such that k+y is odd.
    # This is (x2 - x1 + 1) // 2 if (x1+y) is odd or (x2-1+y) is odd.
    # It is (x2 - x1) // 2 if (x1+y) is even and (x2-1+y) is even.
    
    # Let dx = abs(sx - tx)
    # If (sx + sy) % 2 == 0 and (tx + sy) % 2 == 0 and dx % 2 == 0:
    #    horizontal cost is dx // 2.
    # But we can change y.
    # If we change y by 1, the parity of (x+y) flips.
    # So we can always achieve a horizontal cost of dx // 2.
    # The only question is if we can get (dx-1)//2.
    # If dx is even, (dx+1)//2 and dx//2 are the same.
    # If dx is odd, we can get (dx-1)//2 if there exists y in [sy, ty] 
    # such that (sx+y) is even and (tx+y) is even.
    # But if dx is odd, sx and tx have different parity.
    # So sx+y and tx+y always have different parity.
    # Thus, for any y, one of them is even and one is odd.
    # The number of k in {x1, ..., x2-1} such that k+y is odd is always (dx+1)//2.
    # Wait, let's re-count.
    # If dx=1, x1=3, x2=4. k=3. k+y is 3+y.
    # If y=1, 3+1=4 (even). Cost = 0.
    # If y=0, 3+0=3 (odd). Cost = 1.
    # So if dx=1, cost can be 0 or 1.
    
    # General rule for horizontal cost given y:
    # The boundaries are at k where k+y is odd.
    # The number of boundaries is:
    # (x2 + (y%2)) // 2 - (x1 + (y%2)) // 2
    # We want to minimize this over y in [sy, ty].
    # Let f(y) = (x2 + (y%2)) // 2 - (x1 + (y%2)) // 2.
    # If sy == ty, y is fixed.
    # If sy != ty, we can pick y%2 = 0 or y%2 = 1.
    # So we take min(f(0), f(1)).
    
    # Let's refine:
    # dx = abs(sx - tx)
    # dy = abs(sy - ty)
    # If sy == ty:
    #     cost = (max(sx, tx) + (sy%2)) // 2 - (min(sx, tx) + (sy%2)) // 2
    # Else:
    #     cost = dy + min(
    #         (max(sx, tx) + 0) // 2 - (min(sx, tx) + 0) // 2,
    #         (max(sx, tx) + 1) // 2 - (min(sx, tx) + 1) // 2
    #     )
    # Wait, the vertical moves also cost.
    # Each vertical move from y to y+1 costs 1.
    # But the first vertical move might be "free" if we are already in a tile?
    # No, the rule says "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # If we move from (sx+0.5, sy+0.5) to (sx+0.5, sy+1.5), we enter a new tile. Cost 1.
    # If we move from (sx+0.5, sy+0.5) to (sx+1.5, sy+0.5), we might stay in the same tile.
    
    # Let's use the logic:
    # Total cost = dy + min_y(horizontal_cost(sx, tx, y))
    # But we must be careful: the vertical moves themselves cost.
    # The total cost is dy + min(
    #     (max(sx, tx) + (sy%2)) // 2 - (min(sx, tx) + (sy%2)) // 2,
    #     (max(sx, tx) + (ty%2)) // 2 - (min(sx, tx) + (ty%2)) // 2,
    #     (max(sx, tx) + 0) // 2 - (min(sx, tx) + 0) // 2 if dy > 0 else float('inf'),
    #     (max(sx, tx) + 1) // 2 - (min(sx, tx) + 1) // 2 if dy > 0 else float('inf')
    # )
    # Actually, if dy > 0, we can pick either parity for y.
    # If dy == 0, we are stuck with sy's parity.
    
    # Let's test Sample 1: 5 0, 2 5. dx=3, dy=5.
    # dy > 0, so we take min(f(0), f(1)).
    # f(0) = (5+0)//2 - (2+0