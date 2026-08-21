The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional style, emphasizing expressions over statements. For this problem, the goal is to calculate the minimum toll between two points in a grid of $2\times1$ tiles. 

The tiles are laid out such that if $i+j$ is even, $A_{i,j}$ and $A_{i+1,j}$ are merged. This creates a brick-like pattern. Moving horizontally within a tile is free, but moving vertically always enters a new tile. The optimal strategy is to move horizontally to a column where the tile boundaries allow for "free" horizontal transitions, and then move vertically. Specifically, the cost is determined by the vertical distance and whether the horizontal start and end points fall into the same tile at their respective heights.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The cost to move between two points (sx, sy) and (tx, ty) in this 
    # specific tiling pattern can be derived by observing that:
    # 1. Vertical movement always costs 1 per unit distance.
    # 2. Horizontal movement is free if you are within the same 2x1 tile.
    # 3. The tiles are aligned such that at height y, the boundaries are at 
    #    x = k*2 if y is even, and x = k*2 + 1 if y is odd.
    #    This is equivalent to saying a tile covers x and x+1 if (x+y) is even.
    
    # Let's define the "tile index" for a coordinate (x, y).
    # If (x+y) is even, x and x+1 are in the same tile.
    # We can represent the tile's horizontal identity as:
    # If (x+y) is even, the tile is {x, x+1}. If (x+y) is odd, the tile is {x}.
    # Actually, a simpler observation:
    # The cost is simply the Manhattan distance minus the number of 
    # "free" horizontal steps.
    # A horizontal step from x to x+1 is free if (x+y) is even.
    # This means at any height y, every second horizontal edge is "open".
    
    # The minimum cost is:
    # abs(sy - ty) + (1 if the start and end points are not in the same 
    # horizontal tile alignment relative to the grid's parity else 0)
    # Wait, the above is for 1x1. For 2x1:
    # The cost is abs(sy - ty) + (distance if we must cross a boundary).
    # Let's refine:
    # To move from (sx, sy) to (tx, ty):
    # You must pay abs(sy - ty) for the vertical distance.
    # Additionally, you might pay for horizontal movement.
    # However, you can move horizontally for free if you are in a 2x1 tile.
    # In each row y, you can move between x and x+1 for free if x+y is even.
    # This means in any row, you can move between any x and x' for free 
    # if you can reach the "open" edges.
    # Actually, the rule is: you can move from x to x+1 for free if x+y is even.
    # This means you can move from x to x+2 for free by going x -> x+1 (free) 
    # and then x+1 -> x+2 (costs 1). 
    # No, that's not right. If x+y is even, {x, x+1} is one tile.
    # To move from x to x+2, you must leave the tile {x, x+1} and enter {x+2, x+3} 
    # (if x+2+y is even).
    
    # Correct Logic:
    # In row y, the tiles are [0,1], [2,3], [4,5]... if y is even.
    # In row y, the tiles are [-1,0], [1,2], [3,4]... if y is odd.
    # Let's normalize this: a tile is defined by floor((x + (y % 2)) / 2).
    # Let f(x, y) = (x + (y % 2)) // 2.
    # The cost is abs(sy - ty) + abs(f(sx, sy) - f(tx, ty)).
    # But wait, you can change your f(x, y) by moving vertically.
    # The optimal strategy is to move to the y-level that minimizes the 
    # horizontal distance in terms of tiles.
    # But the vertical cost is constant. 
    # The horizontal cost is minimized if we pick the parity of y that 
    # makes sx and tx fall into the same or adjacent tiles.
    
    # Let's re-evaluate:
    # Cost = abs(sy - ty) + max(0, abs(sx - tx) - (1 if (sx+sy)%2 == 0 and 
    # (tx+ty)%2 == 0 and (sx-tx)%2 == 0 else 0))
    # This is getting complex. Let's use the property:
    # The distance is abs(sy - ty) + abs( (sx + (sy%2))//2 - (tx + (ty%2))//2 )
    # But you can choose to move to sy+1 or ty+1 to potentially reduce the 
    # second term.
    # The second term is the number of tiles crossed horizontally.
    # Let g(x, y) = (x + (y % 2)) // 2.
    # We want to find min(abs(sy - y) + abs(y - ty) + abs(g(sx, sy) - g(tx, y)) + abs(g(tx, y) - g(tx, ty)))
    # This is simplified to:
    # Cost = abs(sy - ty) + abs(g(sx, sy) - g(tx, ty))
    # Because the "zigzag" doesn't help reduce the number of tiles crossed.
    
    # Let's test Sample 1: 5 0 to 2 5
    # g(5, 0) = (5 + 0)//2 = 2
    # g(2, 5) = (2 + 1)//2 = 1
    # Cost = abs(0 - 5) + abs(2 - 1) = 5 + 1 = 6. 
    # Sample 1 output is 5. Why?
    # Because you can move from (5,0) to (4,0) for free (since 4+0 is even).
    # Then g(4, 0) = 2. Then move to (2, 5).
    # Wait, if you are at (5,0), you are in tile {4, 5} because 4+0 is even.
    # So you can move to x=4 for free.
    # At x=4, y=0, you are in tile {4, 5}.
    # Then you move to y=5.
    # At x=4, y=5, you are in tile {4, 5} because 4+5 is odd? No.
    # Let's use the rule: A_{i,j} and A_{i+1,j} are same tile if i+j is even.
    # (5,0) is in A_{5,0}. 5+0 is odd, so A_{5,0} is a standalone tile.
    # (4,0) is in A_{4,0}. 4+0 is even, so A_{4,0} and A_{5,0} are the same tile.
    # Thus, moving from 5.5 to 4.5 at y=0.5 is FREE.
    # Now we are at (4.5, 0.5). We want to reach (2.5, 5.5).
    # Vertical distance is 5.
    # At y=0.5, we are in tile {4, 5}.
    # At y=5.5, we are in tile {2, 3} because 2+5 is odd? No, 2+5=7 (odd).
    # If i+j is even, {i, i+1} is one tile.
    # For y=5, i+j is even if i is odd. So {1,2}, {3,4}, {5,6} are tiles.
    # (2.5, 5.5) is in A_{2,5}. 2+5=7 (odd), so A_{2,5} is a standalone tile.
    # But A_{1,5} and A_{2,5} are NOT the same. A_{3,5} and A_{4,5} ARE the same.
    # So at y=5.5, x=4.5 is in tile {3, 4}.
    # To get from tile {3, 4} to tile {2, 5}, you need 1 move.
    # Total = 5 (vertical) + 0 (horizontal at y=0) + 0 (horizontal at y=5) + 1 (tile transition) = 6?
    # No, the sample says 5. Let's re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # (5.5, 0.5) -> (4.5, 0.5) : Same tile {4, 5}, cost 0.
    # (4.5, 0.5) -> (4.5, 1.5) : Enter tile A_{4,1}, cost 1.
    # (4.5, 1.5) -> (3.5, 1.5) : 3+1=4 (even), so A_{3,1} and A_{4,1} are same tile, cost 0.
    # (3.5, 1.5) -> (3.5, 4.5) : Enter A_{3,2}, A_{3,3}, A_{3,4}, cost 3.
    # (3.5, 4.5) -> (2.5, 4.5) : 2+4=6 (even), so A_{2,4} and A_{3,4} are same tile, cost 0.
    # (2.5, 4.5) -> (2.5, 5.5) : Enter A_{2,5}, cost 1.
    # Total = 0 + 1 + 0 + 3 + 0 + 1 = 5.
    
    # The pattern is: you can move horizontally for free if you are in a 2x1 tile.
    # This means you can change your x-coordinate by 1 for free if (x+y) is even.
    # This looks like a shortest path on a graph where nodes are tiles.
    # The distance is abs(sy - ty) + abs(g(sx, sy) - g(tx, ty)) 
    # where g(x, y) is the index of the tile in row y.
    # g(x, y) = x // 2 if y is even else (x + 1) // 2.
    # Wait, if y is even, tiles are {0,1}, {2,3}... so g(x, y) = x // 2.
    # If y is odd, tiles are {-1,0}, {1,2}, {3,4}... so g(x, y) = (x + 1) // 2.
    # Let's check Sample 1: sx=5, sy=0, tx=2, ty=5.
    # g(5, 0) = 5 // 2 = 2.
    # g(2, 5) = (2 + 1) // 2 = 1.
    # Cost = abs(0 - 5) + abs(2 - 1) = 5 + 1 = 6. Still 6.
    # What's wrong? The vertical moves also enter new tiles.
    # The vertical move from (x, y) to (x, y+1) enters tile A_{x, y+1}.
    # If A_{x, y+1} is the same tile as A_{x-1, y+1}, it doesn't matter.
    # The key is that you can "shift" your x-coordinate for free.
    # In row y, you can reach any x' such that g(x, y) == g(x', y).
    # The cost is the number of tiles visited.
    # This is a BFS problem on tiles.
    # The distance between tile (g1, y1) and (g2