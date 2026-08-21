```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1 or 1x2.
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile (Horizontal).
    # If i+j is odd, A_{i,j} and A_{i,j+1} are one tile (Vertical).
    # This creates a checkerboard of horizontal and vertical tiles.
    
    # Let's analyze the cost to move between cells.
    # Moving from (i, j) to (i+1, j):
    # If i+j is even, they are in the same tile. Cost = 0.
    # If i+j is odd, they are in different tiles. Cost = 1.
    
    # Moving from (i, j) to (i, j+1):
    # If i+j is odd, they are in the same tile. Cost = 0.
    # If i+j is even, they are in different tiles. Cost = 1.
    
    # Notice that the cost to move 1 unit is 1 if (i+j) has the "wrong" parity.
    # Specifically, the cost to move from (sx, sy) to (tx, ty) is:
    # sum_{i=sx}^{tx-1} [i+sy is odd] + sum_{j=sy}^{ty-1} [tx+j is even]
    # However, the path can be optimized. 
    # The cost is actually the distance in a graph where edges have weights 0 or 1.
    # This specific tiling is a known problem where the distance is:
    # dist = max(abs(sx-tx), abs(sy-ty), (abs(sx-tx) + abs(sy-ty) + 1) // 2)
    # Wait, that's for a different tiling. Let's re-evaluate.
    
    # Let's use the property: 
    # Cost to move (i, j) -> (i+1, j) is 0 if i+j even, 1 if i+j odd.
    # Cost to move (i, j) -> (i, j+1) is 0 if i+j odd, 1 if i+j even.
    
    # This is equivalent to saying:
    # You pay 1 if you cross a boundary that isn't the "open" side of the tile.
    # In this tiling, every cell (i, j) is part of a 2x1 block.
    # If i+j is even, the block is {(i, j), (i+1, j)}.
    # If i+j is odd, the block is {(i, j), (i, j+1)}.
    
    # Let's transform coordinates: 
    # The cost is simply the Manhattan distance divided by 2, rounded up,
    # but it depends on the starting parity.
    # Actually, the minimum toll is simply:
    # (abs(sx - tx) + abs(sy - ty) + (1 if (sx + sy) % 2 == (tx + ty) % 2 else 0)) // 2
    # Let's test Sample 1: 5 0 to 2 5. 
    # abs(5-2) + abs(0-5) = 3 + 5 = 8. 
    # (5+0)%2 = 1, (2+5)%2 = 1. Parity is same.
    # (8 + 1) // 2 = 4. But sample output says 5.
    
    # Re-evaluating:
    # The cost to move from (i, j) to (i+1, j) is (i + j) % 2.
    # The cost to move from (i, j) to (i, j+1) is (i + j + 1) % 2.
    # This is a shortest path problem on a grid.
    # The cost is simply:
    # If we move from (sx, sy) to (tx, ty), the total cost is:
    # sum_{i=min(sx,tx)}^{max(sx,tx)-1} (i + sy) % 2  +  sum_{j=min(sy,ty)}^{max(sy,ty)-1} (tx + j + 1) % 2
    # But we can choose the path. The optimal path is to move in a way that we 
    # utilize the 0-cost edges.
    # The 0-cost edges are: (i, j) -> (i+1, j) if i+j is even, and (i, j) -> (i, j+1) if i+j is odd.
    # This means we can move for free if we follow the pattern: 
    # (even, even) -> (odd, even) -> (odd, odd) -> (even, odd) -> (even, even)
    # This is a cycle of length 4 with total cost 0? No.
    # Let's trace:
    # (0,0) --0--> (1,0) --1--> (1,1) --0--> (1,2) --1--> (0,2) --0--> (0,1) --1--> (0,0)
    # Wait, the 0-cost edges are:
    # If i+j is even: (i, j) <-> (i+1, j)
    # If i+j is odd: (i, j) <-> (i, j+1)
    # This means from (0,0) we can go to (1,0) for free.
    # From (1,0), i+j=1 (odd), so we can go to (1,1) for free.
    # From (1,1), i+j=2 (even), so we can go to (2,1) or (0,1) for free.
    # From (2,1), i+j=3 (odd), so we can go to (2,2) or (2,0) for free.
    # Essentially, we can move in "L" shapes of 2 units for free.
    # The distance is simply the Manhattan distance divided by 2, rounded up.
    # Let's check Sample 1: (5,0) to (2,5). Dist = 3 + 5 = 8. 8/2 = 4. Still not 5.
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are contained in the same tile."
    # This means the edge between (i, j) and (i+1, j) is FREE if i+j is even.
    # Otherwise, the edge between (i, j) and (i, j+1) is FREE if i+j is odd.
    # Let's map the costs:
    # Horizontal edge ((i, j), (i+1, j)): cost 0 if i+j even, 1 if i+j odd.
    # Vertical edge ((i, j), (i, j+1)): cost 0 if i+j odd, 1 if i+j even.
    
    # This is a distance problem on a graph. Since the costs are 0 and 1, 
    # we can use the fact that the cost to move from (sx, sy) to (tx, ty) is:
    # cost = (abs(sx - tx) + abs(sy - ty) + (1 if (sx + sy) % 2 == (tx + ty) % 2 else 0)) // 2
    # Wait, I just tried that and it gave 4. Let me re-calculate.
    # Sample 1: sx=5, sy=0, tx=2, ty=5.
    # abs(5-2) = 3, abs(0-5) = 5. Sum = 8.
    # (5+0)%2 = 1. (2+5)%2 = 1. 
    # They have the same parity. So (8 + 1) // 2 = 4.
    # But the answer is 5. Why?
    # Let's trace the path: (5,0) -> (4,0) [i=4, j=0, i+j=4 even: FREE]
    # (4,0) -> (3,0) [i=3, j=0, i+j=3 odd: COST 1]
    # (3,0) -> (2,0) [i=2, j=0, i+j=2 even: FREE]
    # Now at (2,0). Need to go to (2,5).
    # (2,0) -> (2,1) [i=2, j=0, i+j=2 even: COST 1]
    # (2,1) -> (2,2) [i=2, j=1, i+j=3 odd: FREE]
    # (2,2) -> (2,3) [i=2, j=2, i+j=4 even: COST 1]
    # (2,3) -> (2,4) [i=2, j=3, i+j=5 odd: FREE]
    # (2,4) -> (2,5) [i=2, j=4, i+j=6 even: COST 1]
    # Total cost = 1 + 1 + 1 + 1 = 4.
    # Wait, the sample says 5. Let me re-read again.
    # "Each time he enters a tile, he pays a toll of 1."
    # This means the starting tile is already paid? No, "enters a tile".
    # If he starts in tile T1 and moves to T2, he pays 1.
    # If he moves within T1, he pays 0.
    # Sample 1: (5,0) is in tile A_{4,0}U A_{5,0} (since 4+0 is even).
    # (2,5) is in tile A_{2,5}U A_{3,5} (since 2+5 is odd... wait).
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # For (2,5): i=2, j=5. i+j=7 (odd). 
    # If i+j is odd, A_{i,j} and A_{i,j+1} are one tile.
    # So A_{2,5} and A_{2,6} are one tile.
    # The target is (2.5, 5.5), which is in A_{2,5}.
    # The start is (5.5, 0.5), which is in A_{5,0}.
    # A_{5,0}: i=5, j=0. i+j=5 (odd). So A_{5,0} and A_{5,1} are one tile.
    
    # Let's use the coordinate transformation:
    # A cell (i, j) belongs to tile:
    # If i+j is even, it's tile {(i, j), (i+1, j)} if i is even, or {(i-1, j), (i, j)} if i is odd.
    # Actually, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j, the cells (i, j) are paired as (0,j)-(1,j), (2,j)-(3,j)... if j is even.
    # And (1,j)-(2,j), (3,j)-(4,j)... if j is odd.
    
    # This is exactly the distance in a graph where:
    # Edge ((i,j), (i+1,j)) has weight 0 if i+j is even, 1 if i+j is odd.
    # Edge ((i,j), (i,j+1)) has weight 0 if i+j is odd, 1 if i+j is even.
    # The distance between (sx, sy) and (tx, ty) in such a graph is:
    # dist = abs(sx-tx) + abs(sy-ty) - (number of 0-weight edges on the path)
    # To maximize 0-weight edges, we can take at most one 0-weight edge per unit of Manhattan distance.
    # But we can't have two 0-weight edges in a row because they are perpendicular.
    # The maximum number of 0-weight edges is (abs(sx-tx) + abs(sy-ty) + 1) // 2.
    # However, this is only if the