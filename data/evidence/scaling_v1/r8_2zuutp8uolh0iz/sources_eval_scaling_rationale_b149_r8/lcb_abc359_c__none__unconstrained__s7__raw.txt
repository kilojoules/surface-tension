The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing expressions over statements. In this specific problem, since the optimal cost is determined by a closed-form mathematical formula based on the coordinates, this constraint encourages the use of direct calculation and built-in functions rather than iterative search or dynamic programming.

To solve this, I will use `abs()` for distances and `max()` to handle the cost logic, implementing the solution as a single expression within a `print()` statement.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map inputs to variables
    # Sx, Sy: Start coordinates
    # Tx, Ty: Target coordinates
    coords = list(map(int, input_data))
    Sx, Sy, Tx, Ty = coords[0], coords[1], coords[2], coords[3]

    # The tiling pattern:
    # A tile covers A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j, tiles are pairs (0,1), (2,3), etc. if j is even,
    # and pairs (1,2), (3,4), etc. if j is odd.
    # This is a Manhattan-like distance problem on a graph where some edges have weight 0.
    # The cost to move between (Sx, Sy) and (Tx, Ty) can be derived as:
    # Let dx = |Sx - Tx| and dy = |Sy - Ty|.
    # The cost is essentially the number of tile boundaries crossed.
    # Specifically, moving vertically always costs 1 per unit.
    # Moving horizontally costs 1 only if you cross a boundary between two tiles.
    # The optimal strategy is to move diagonally in a sense.
    # The formula for the minimum toll is max(abs(Sx - Tx), abs(Sy - Ty), (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2)
    # Wait, the actual logic for this specific tiling is:
    # Cost = max(|Sy - Ty|, (|Sx - Tx| + |Sy - Ty| + 1) // 2) if we consider the parity.
    # Actually, the simplest form for this specific grid is:
    # Let dx = abs(Sx - Tx), dy = abs(Sy - Ty)
    # The cost is max(dy, (dx + dy + 1) // 2) is for a different grid.
    # For this grid: the cost is max(dy, (dx + dy) // 2) if we can move 'for free' 
    # horizontally every other step.
    # The correct formula for this problem is:
    # cost = max(abs(Sy - Ty), (abs(Sx - Tx) + abs(Sy - Ty)) // 2)
    # But we must account for the starting tile's orientation.
    # Let's refine: the distance is max(|Sy - Ty|, ceil((|Sx - Tx| + |Sy - Ty|) / 2))
    # if we are allowed to move horizontally for free.
    # However, the parity of i+j determines the tile.
    # The distance is actually simpler: max(abs(Sy - Ty), (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2)
    # is for a different problem. For this one:
    # The cost is max(abs(Sy - Ty), (abs(Sx - Tx) + abs(Sy - Ty)) // 2) 
    # is almost correct, but we need to check the parity of the start and end.
    # Let's use the property: cost = max(dy, (dx + dy + 1) // 2) is for the 
    # case where you can move 2 units horizontally for the price of 1 vertical.
    # The correct formula for this specific tiling is:
    # ans = max(abs(Sy - Ty), (abs(Sx - Tx) + abs(Sy - Ty) + 1) // 2)
    # Let's double check Sample 1: 5 0 to 2 5. dx=3, dy=5. max(5, (3+5+1)//2) = 5. Correct.
    # Sample 2: 3 1 to 4 1. dx=1, dy=0. max(0, (1+0+1)//2) = 1? No, Sample 2 says 0.
    # The condition for 0 cost in Sample 2: (3,1) and (4,1). i=3, j=1. i+j = 4 (even).
    # Rule: if i+j is even, A_{i,j} and A_{i+1,j} are same tile.
    # Here 3+1=4 is even, so A_{3,1} and A_{4,1} are the same tile. Cost 0.
    # The general formula is: 
    # Let dx = abs(Sx - Tx), dy = abs(Sy - Ty)
    # If we are in the same tile, cost is 0.
    # The cost is max(dy, (dx + dy + (1 if (Sx+Sy)%2 != (Tx+Ty)%2 else 0)) // 2) 
    # Actually, the most reliable formula for this grid is:
    # cost = max(abs(Sy - Ty), (abs(Sx - Tx) + abs(Sy - Ty) + (1 if (Sx+Sy)%2 != (Tx+Ty)%2 else 0)) // 2)
    # Wait, let's re-evaluate:
    # In one step (cost 1), you can move from (x, y) to (x+1, y+1), (x-1, y+1), (x, y+1), etc.
    # The distance is max(|Sy - Ty|, ceil((|Sx - Tx| + |Sy - Ty|) / 2))
    # Let's check Sample 2 again: dx=1, dy=0. (1+0)/2 = 0.5. Ceil is 1. Still not 0.
    # The parity of the tile matters. 
    # A point (x, y) is in tile T if (x+y) is even and it's in {x, x+1} or (x+y) is odd and it's in {x-1, x}.
    # Let's use the coordinate transformation: 
    # A point (x, y) belongs to a tile identified by ( (x + (x+y)%2) // 2, y )
    # Let X = (x + (x+y)%2) // 2, Y = y.
    # The distance between (X1, Y1) and (X2, Y2) in this transformed grid 
    # where you can move to any 8-neighbor is max(|X1-X2|, |Y1-Y2|).
    # But we can only move in 4 directions.
    # The cost to move from (X1, Y1) to (X2, Y2) is |Y1-Y2| + max(0, |X1-X2| - |Y1-Y2|)
    # if we can move diagonally. But we can't.
    # Actually, the cost is simply:
    # dx = abs(Sx - Tx), dy = abs(Sy - Ty)
    # The answer is max(dy, (dx + dy + 1) // 2) if we consider the parity.
    # Let's use the transformation:
    # X = (x + (x+y)%2) // 2, Y = y
    # The distance is |Y1 - Y2| + max(0, |X1 - X2| - |Y1 - Y2|) if we can move diagonally.
    # With only 4-direction moves, the cost is:
    # If we move from (X1, Y1) to (X2, Y2), the cost is |Y1 - Y2| + |X1 - X2| 
    # BUT we can move diagonally by combining a horizontal and vertical move.
    # One vertical move (cost 1) and one horizontal move (cost 0 or 1).
    # The correct distance is:
    # X1 = (Sx + (Sx + Sy) % 2) // 2
    # Y1 = Sy
    # X2 = (Tx + (Tx + Ty) % 2) // 2
    # Y2 = Ty
    # ans = max(abs(Y1 - Y2), (abs(X1 - X2) + abs(Y1 - Y2) + 1) // 2) 
    # No, the simplest form is:
    # ans = max(abs(Y1 - Y2), abs(X1 - X2)) 
    # Let's check Sample 1: Sx=5, Sy=0, Tx=2, Ty=5
    # X1 = (5 + (5+0)%2)//2 = (5+1)//2 = 3; Y1 = 0
    # X2 = (2 + (2+5)%2)//2 = (2+1)//2 = 1; Y2 = 5
    # max(|3-1|, |0-5|) = max(2, 5) = 5. Correct.
    # Sample 2: Sx=3, Sy=1, Tx=4, Ty=1
    # X1 = (3 + (3+1)%2)//2 = (3+0)//2 = 1; Y1 = 1
    # X2 = (4 + (4+1)%2)//2 = (4+1)//2 = 2; Y2 = 1
    # Wait, X1=1, X2=2. max(0, 1) = 1. Still not 0.
    # Let's re-calculate X for Sample 2:
    # Sx=3, Sy=1 => (3+1)%2 = 0. X1 = (3+0)//2 = 1.
    # Tx=4, Ty=1 => (4+1)%2 = 1. X2 = (4+1)//2 = 2.
    # There is a mistake in my X calculation.
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # For j=1: i+1 is even => i is odd. So A_{1,1}&A_{2,1}, A_{3,1}&A_{4,1} are tiles.
    # (3,1) and (4,1) are both in A_{3,1} and A_{4,1} respectively.
    # So they are in the same tile.
    # The tile index for A_{i,j} is:
    # If (i+j)%2 == 0: tile_x = i // 2 if j%2==0 else (i+1) // 2
    # This is getting complex. Let's use:
    # Tile ID for A_{i,j} is ( (i + (i+j)%2) // 2, j )
    # Sample 2: A_{3,1} -> (3 + (3+1)%2)//2 = 3//2 = 1. A_{4,1} -> (4 + (4+1)%2)//2 = 5//2 = 2.
    # Still different. Let's try: Tile ID is ( (i - (i+j)%2) // 2, j )
    # Sample 2: A_{3,1} -> (3 - 0)//2 = 1. A_{4,1} -> (4 - 1)//2 = 1. MATCH!
    # Sample 1: A_{5,0} -> (5 - (5+0)%2)//2 = (5-1)//2 = 2. Y=0.
    # A_{2,5} -> (2 - (2+5)%2)//2 = (2-1)//2 = 0. Y=5.
    # max(|2-0|, |0-5|) = 5. MATCH!
    # Final Formula: X = (i - (i+j)%2) // 2, Y = j. Ans = max(abs(X1-X2), abs(Y1-Y2))
    # Wait, the distance in a grid where you can move to any 8-neighbor is max(dx, dy).
    # But we can only move in 4 directions. However, the "cost" is what matters.
    # Moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1.
    # But moving from (X, Y) to (X+1, Y+1) also costs 1 (one vertical move).
    # So it is indeed the Chebyshev distance: max(|X1-X2|, |Y1-Y2|).

    print(max(abs((Sx - (Sx + Sy) % 2)