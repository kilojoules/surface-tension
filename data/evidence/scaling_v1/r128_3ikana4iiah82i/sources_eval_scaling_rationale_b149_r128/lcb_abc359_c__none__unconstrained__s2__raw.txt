The constraint to avoid explicit loops and recursion forces a shift toward functional programming paradigms, which can lead to more concise, expression-oriented code but often departs from the imperative style typical of Python. To solve this problem, I will use arithmetic expressions to calculate the distance and a conditional expression (ternary operator) to handle the logic, ensuring the solution remains a series of assignments and a final print statement.

```python
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
    # This means tiles are 2x1 rectangles aligned horizontally.
    # The boundary between tiles occurs at x = k (where k is even) when j is even,
    # and at x = k (where k is odd) when j is odd.
    # Essentially, a tile boundary exists at x = k if (k + j) is odd.
    
    # To move from (sx, sy) to (tx, ty), we can move horizontally then vertically.
    # Moving vertically across j-boundaries always costs 1 per unit because
    # every A_{i,j} and A_{i,j+1} are in different tiles.
    # Moving horizontally costs 1 only if we cross a boundary.
    
    # Let's analyze the cost:
    # Vertical cost: abs(sy - ty)
    # Horizontal cost: 
    # If we are at height j, we cross a boundary at x=k if (k+j) is odd.
    # In a span of width W, we cross W/2 boundaries.
    # Specifically, if we move from sx to tx at height j:
    # The number of boundaries crossed is the number of k between sx and tx 
    # such that (k+j) is odd.
    
    # However, we can choose the height j at which we move horizontally.
    # If we move horizontally at height j, the cost is:
    # (abs(sx - tx) + 1) // 2 if (sx+j) and (tx+j) have different parity 
    # relative to the boundary condition, etc.
    # Actually, for any two x-coordinates sx and tx, there exists a height j
    # such that the number of boundaries crossed is floor(abs(sx-tx)/2).
    # Specifically, if we pick j such that (sx+j) is even, the boundary is at 
    # k+j = odd. The boundaries are at k = 1-j, 3-j, etc.
    # The number of integers k in (min(sx, tx), max(sx, tx)] such that k+j is odd
    # is always abs(sx-tx)//2 if we pick j optimally.
    
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means the "gap" (boundary) is between A_{i,j} and A_{i+1,j} when i+j is odd.
    # For a fixed j, the boundaries are at x = i+1 where i+j is odd.
    # This means boundaries are at x = k where k+j is even.
    # Number of k in (min(sx, tx), max(sx, tx)] such that k+j is even:
    # This is abs(sx-tx)//2 if we pick j such that the endpoints are not boundaries.
    # If we can pick any j along the path, we can minimize this.
    # The vertical cost is fixed at abs(sy-ty).
    # The horizontal cost is minimized when we pick j such that we cross the fewest boundaries.
    # For any sx, tx, the minimum number of boundaries crossed is abs(sx-tx)//2.
    # This is achieved by picking j such that (sx+j) is even (then the first step 
    # from sx to sx+1 is free).
    
    # Total cost = abs(sy - ty) + abs(sx - tx) // 2
    # But we must check if the starting and ending tiles are the same.
    # The problem asks for the toll paid when ENTERING a tile.
    # Starting tile is free.
    
    # Let's re-evaluate:
    # Vertical movement: Each step from j to j+1 enters a new tile. Cost: abs(sy-ty).
    # Horizontal movement: At height j, boundaries are at x=k where k+j is even.
    # To get from sx to tx, we cross k boundaries.
    # If we pick j such that (sx+j) is even, the boundary is at x=sx+1, sx+3...
    # Wait, if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # So the boundary is at x=k where (k-1)+j is odd, i.e., k+j is even.
    # If we pick j such that sx+j is even, then A_{sx,j} and A_{sx+1,j} are one tile.
    # The first boundary we hit is at x=sx+2 (since (sx+1)+j is odd).
    # The number of boundaries crossed is abs(sx-tx) // 2.
    
    # Example 1: (5,0) to (2,5). 
    # abs(0-5) + abs(5-2)//2 = 5 + 3//2 = 5 + 1 = 6. 
    # But sample output says 5. Let's re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is already entered.
    # If we move from (5,0) to (4,0), and 4+0 is even, A_{4,0} and A_{5,0} are one tile.
    # So moving 5 -> 4 costs 0.
    # Then move (4,0) to (4,5). This crosses 5 boundaries. Cost 5.
    # Then move (4,5) to (2,5). 
    # At j=5, boundaries are at x=k where k+5 is even (k is odd).
    # x=3 is a boundary, x=4 is not.
    # Moving 4 -> 3 crosses a boundary (3+5=8 is even). Cost 1.
    # Moving 3 -> 2 is free (since 2+5=7 is odd, A_{2,5} and A_{3,5} are one tile).
    # Total = 0 + 5 + 1 = 6. Still not 5.
    
    # Let's re-examine:
    # At j=0, boundaries are at x=k where k is even.
    # sx=5, tx=2. Boundaries at x=4, 2.
    # If we move 5 -> 4 (crosses x=4, cost 1), 4 -> 3 (free), 3 -> 2 (crosses x=2, cost 1).
    # But if we move 5 -> 6 (free), 6 -> 7... no.
    # If we move 5 -> 4 at j=0, cost is 1.
    # If we move 5 -> 4 at j=1, 4+1=5 (odd), so A_{4,1} and A_{5,1} are one tile. Cost 0!
    # So at j=1, moving 5 -> 4 is free.
    # At j=1, moving 4 -> 3 costs 1 (3+1=4 even).
    # At j=1, moving 3 -> 2 is free (2+1=3 odd).
    # So horizontal cost is abs(sx-tx)//2.
    # Total = abs(sy-ty) + abs(sx-tx)//2.
    # For Sample 1: abs(0-5) + abs(5-2)//2 = 5 + 1 = 6.
    # Wait, the sample says 5. How?
    # "Move left by 1 (5->4). Pay 0." -> This means at j=0, 5->4 is free.
    # If 5->4 is free at j=0, then A_{4,0} and A_{5,0} are one tile.
    # Rule: "When i+j is even, A_{i,j} and A_{i+1,j} are one tile."
    # For i=4, j=0: 4+0=4 (even). So A_{4,0} and A_{5,0} are one tile. Correct.
    # Then "Move up by 1 (0->1). Pay 1."
    # Then "Move left by 1 (4->3). Pay 0."
    # For i=3, j=1: 3+1=4 (even). So A_{3,1} and A_{4,1} are one tile. Correct.
    # Then "Move up by 3 (1->4). Pay 3."
    # Then "Move left by 1 (3->2). Pay 0."
    # For i=2, j=4: 2+4=6 (even). So A_{2,4} and A_{3,4} are one tile. Correct.
    # Then "Move up by 1 (4->5). Pay 1."
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # Analysis:
    # We can move horizontally for free if we are at a height j where i+j is even.
    # To move from sx to tx, we need to cross abs(sx-tx) units.
    # Each unit is either free or costs 1.
    # But we can change j to make the current horizontal step free!
    # However, changing j costs 1.
    # This looks like a shortest path on a graph.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # The vertical distance is mandatory: abs(sy - ty).
    # The horizontal distance can be covered by "free" steps if we are at the right parity of j.
    # For any x, there is one parity of j that makes the step x -> x+1 free.
    # If we are already at that parity, cost 0. If not, we must have moved vertically.
    # Actually, the minimum cost is simply:
    # max(abs(sy - ty), (abs(sx - tx) + 1) // 2) is not correct.
    # Let's look at the parity.
    # To move x -> x+1, we need j such that x+j is even.
    # To move x+1 -> x+2, we need j such that x+1+j is even.
    # These two require different parities of j.
    # So every 2 units of horizontal movement requires at least one change in j-parity.
    # A change in j-parity costs 1 (a vertical move).
    # So the cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2).
    # Let's check Sample 1: max(5, (3+1)//2) = max(5, 2) = 5. Correct!
    # Sample 2: (3,1) to (4,1). max(0, (1+1)//2) = 1. 
    # Wait, Sample 2 output is 0.
    # In Sample 2, sx=3, tx=4, sy=1, ty=1.
    # i=3, j=1: i+j = 4 (even). So A_{3,1} and A_{4,1} are one tile.
    # Moving from 3.5 to 4.5 is free. Cost 0.
    # My formula gives 1. The issue is the parity of the starting position.
    # If (sx + sy) is even, the first step (sx -> sx+1) is free.
    # If (sx + sy) is odd, the first step (sx -> sx-1) is free.
    # Let's use the property: a step from x to x+1 is free if x+j is even.
    # A step from x to x-1 is free if (x-1)+j is even.
    # In Sample 2: sx=3, sy=1. sx+sy = 4 (even). 
    # Step 3 -> 4 is free because 3+1=4 is even.
    # In Sample 1: sx=5, sy=0. sx+sy = 5 (odd).
    # Step 5 -> 4 is free because 4+0=4 is even.
    # So if we move in