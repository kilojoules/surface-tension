```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)
    
    # The tiling pattern is:
    # If (i + j) is even, (i, j) and (i+1, j) are one tile.
    # This means for a fixed j, the tiles are horizontal pairs starting at even (i+j).
    # Let's transform the coordinates to make the grid uniform.
    # Consider a coordinate system where each "unit" is a 2x2 block.
    # However, a simpler way is to observe the Manhattan distance and the parity.
    
    # Let dx = |sx - tx| and dy = |sy - ty|.
    # Moving vertically (changing y) always enters a new tile because tiles are 2x1 (horizontal).
    # Moving horizontally (changing x) might stay in the same tile or enter a new one.
    
    # Specifically, if we are at (x, y) and move to (x+1, y):
    # We stay in the same tile if (x + y) is even.
    # We enter a new tile if (x + y) is odd.
    
    # To minimize tolls, we want to maximize the number of horizontal moves 
    # that occur when (x + y) is even.
    
    # Let's normalize the problem:
    # The cost to move from (sx, sy) to (tx, ty) is:
    # Cost = dy + (number of horizontal boundaries crossed that are not "internal" to a tile)
    
    # A horizontal boundary is between x and x+1. It is internal if (x + y) is even.
    # For a fixed y, as x goes from min(sx, tx) to max(sx, tx) - 1,
    # the number of x such that (x + y) is odd is ceil(dx / 2) or floor(dx / 2).
    
    # However, we can pick WHICH y to perform the horizontal movement at.
    # If we move horizontally at y, the cost is the number of x in [min(sx, tx), max(sx, tx)-1]
    # such that (x + y) is odd.
    # This count is dx // 2 if dx is even, and (dx // 2) or (dx // 2 + 1) if dx is odd.
    # We can choose y to be either sy or ty (or anything in between) to minimize this.
    
    # If dx is even, the cost is dx // 2 regardless of y.
    # If dx is odd, the cost is dx // 2 if we can find a y such that 
    # (min(sx, tx) + y) is even, otherwise it is dx // 2 + 1.
    # Since we can pick any y between sy and ty, we can get dx // 2 if 
    # there exists y in [min(sy, ty), max(sy, ty)] such that (min(sx, tx) + y) is even.
    # This is always possible unless sy == ty and (min(sx, tx) + sy) is odd.
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    
    # Minimum horizontal cost:
    # If dx == 0, h_cost = 0.
    # If dx > 0:
    #   If sy != ty, we can pick y to make the first step "free", so h_cost = (dx + 1) // 2 is not necessarily true.
    #   Wait, if sy != ty, we can move to a y where (x + y) is even and then move horizontally.
    #   Actually, if sy != ty, we can always achieve h_cost = dx // 2.
    #   If sy == ty, h_cost = (dx + 1) // 2 if (min(sx, tx) + sy) is odd, else dx // 2.
    
    # Let's re-evaluate:
    # To get from sx to tx, we must cross dx boundaries.
    # Boundary x is "free" if x + y is even.
    # For a fixed y, the boundaries are x, x+1, ...
    # The parities of (x+y) alternate.
    # In any sequence of dx boundaries, at least dx // 2 have (x+y) odd.
    # If dx is even, exactly dx // 2 are odd.
    # If dx is odd, either dx // 2 or (dx // 2 + 1) are odd.
    # We can pick y to make it dx // 2 if we have at least two different y values to choose from (sy and sy+1).
    
    # Correct Logic:
    # Total Cost = dy + (cost to move dx horizontally)
    # If sy == ty:
    #    h_cost = (dx + 1) // 2 if (min(sx, tx) + sy) % 2 != 0 else dx // 2
    # If sy != ty:
    #    h_cost = dx // 2 (because we can pick y or y+1 to ensure the "extra" boundary is even)
    
    # But wait, we can move horizontally at any y between sy and ty.
    # If sy != ty, we have both an even and an odd y available.
    # One of them will make the sequence of dx boundaries start with an even (free) boundary.
    # Thus, if sy != ty, h_cost = dx // 2.
    # If sy == ty, h_cost = (dx + 1) // 2 if (min(sx, tx) + sy) % 2 != 0 else dx // 2.
    
    # Let's check Sample 1: 5 0 -> 2 5. dx=3, dy=5. sy!=ty, so h_cost = 3//2 = 1. Total = 5 + 1 = 6? 
    # Sample 1 output is 5. Let me re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    
    # Let's use the property:
    # A move is free if it doesn't change the tile.
    # Horizontal move (x, y) -> (x+1, y) is free if x+y is even.
    # Vertical move (x, y) -> (x, y+1) is always paid.
    
    # To minimize:
    # 1. Move vertically to a y that is "favorable" for horizontal movement.
    # 2. Move horizontally.
    # 3. Move vertically to the destination.
    
    # If we move horizontally at height y, the cost is the number of x in [min(sx, tx), max(sx, tx)-1]
    # such that x + y is odd.
    # This count is:
    # If dx is even: dx // 2
    # If dx is odd: (dx // 2) if (min(sx, tx) + y) is even, else (dx // 2 + 1)
    
    # We want to minimize this over y in [min(sy, ty), max(sy, ty)].
    # If sy != ty, we can pick y such that (min(sx, tx) + y) is even.
    # Then h_cost = dx // 2.
    # If sy == ty, h_cost = (dx + 1) // 2 if (min(sx, tx) + sy) % 2 != 0 else dx // 2.
    
    # Total cost = dy + h_cost.
    # Sample 1: 5 0, 2 5. dx=3, dy=5. sy!=ty => h_cost = 3//2 = 1. Total = 5 + 1 = 6.
    # Still 6. Why is it 5?
    # Ah, the sample says: "Move left 1 (free), Up 1 (paid), Left 1 (free), Up 3 (paid), Left 1 (free), Up 1 (paid)".
    # Total paid: 1 + 3 + 1 = 5.
    # In this path, he interleaved horizontal and vertical moves.
    # He moved horizontally at y=0, y=1, and y=4.
    # At y=0, x=4: 4+0=4 (even) -> free.
    # At y=1, x=3: 3+1=4 (even) -> free.
    # At y=4, x=2: 2+4=6 (even) -> free.
    # Total cost = dy + 0 = 5.
    
    # If we can interleave, we can potentially make every horizontal move free.
    # A horizontal move at (x, y) is free if x+y is even.
    # We can change the parity of y by moving vertically.
    # If we have dy > 0, we can always pick the parity of y to make the next horizontal move free.
    # The only constraint is that we must move vertically dy times.
    # Each vertical move costs 1.
    # If we have dy > 0, we can distribute the dx horizontal moves across different y levels.
    # Since we can move n units, we can move 1 unit horizontally, 1 unit vertically, 1 unit horizontally...
    # This way, we can make every single horizontal move free?
    # No, because we only have dy vertical moves.
    # But we can move n units.
    # If we are at (x, y) and want to move to (x-1, y), it's free if x-1+y is even.
    # If x-1+y is odd, we can move to (x, y+1) first (cost 1), then (x-1, y+1) (free if x-1+y+1 is even).
    
    # Let's reconsider:
    # To move from (sx, sy) to (tx, ty), we must change x by dx and y by dy.
    # Each vertical step costs 1. Total vertical cost = dy.
    # For each horizontal step, we can choose to do it at any y between sy and ty.
    # If we can find a y such that x+y is even for all x we cross, that's impossible.
    # But we can split dx into several parts and move them at different y.
    # For any x, we can move from x to x+1 for free if we are at a y such that x+y is even.
    # Since we can move to any y in [min(sy, ty), max(sy, ty)], we can cross the boundary x|x+1
    # for free if there is at least one y in that range such that x+y is even.
    # If sy != ty, the range [min(sy, ty), max(sy, ty)] contains both an even and an odd integer.
    # Thus, for any x, we can find a y in the range such that x+y is even.
    # This means if sy != ty, all horizontal moves can be free.
    # Total cost = dy.
    # If sy == ty, we are stuck with one y. The cost is dy + h_cost (where h_cost is calculated above).
    
    # Sample 1: 5 0, 2 5. sy=0, ty=5. sy != ty. Cost = dy = 5. Correct.
    # Sample 2: 3 1, 4 1. sy=1, ty=1. sy == ty. dx=1, dy=0. 
    # min(sx, tx)=3, sy=1. 3+1=4 (even). h_cost = 1 // 2 = 0. Total = 0. Correct.
    
    # Final Logic:
    # if sy != ty:
    #     ans = dy
    # else:
    #     # sy == ty
    #     # h_cost = number of x in [min(sx, tx), max(sx, tx)-1] such that x + sy is odd
    #     # Let L = min(sx, tx), R = max(sx, tx)
    #     # We count x in {L, L+1, ..., R-1} where x % 2 != sy % 2
    #     # This is (R - L + 1) // 2 if (R-L) is even
    #     # If (R-L) is odd, it's either (R-L)//2 or (R-L)//2 + 1
    #     # Let's use: count_odds(N) = (N + 1) // 2
    #     # Count of x in [0, N-1] with x % 2 != p
    #     # If p=0, count odds in [0, N-1] = N // 2
    #     # If p=1, count evens in