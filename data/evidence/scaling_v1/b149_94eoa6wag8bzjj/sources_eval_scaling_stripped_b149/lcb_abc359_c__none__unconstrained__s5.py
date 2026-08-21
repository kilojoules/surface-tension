```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiles are 2x1 rectangles.
    # Rule: If i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j, the tiles are paired as (0,1), (2,3), (4,5)... if j is even
    # and (-1,0), (1,2), (3,4)... if j is odd.
    
    # Let's define a coordinate transformation to a grid where each cell is a tile.
    # For a cell (i, j), its tile ID can be represented as (u, v):
    # If j is even:
    #   The pair (2k, j) and (2k+1, j) belong to tile (k, j).
    #   u = i // 2, v = j
    # If j is odd:
    #   The pair (2k-1, j) and (2k, j) belong to tile (k, j).
    #   u = (i + 1) // 2, v = j
    
    # More simply:
    # If j % 2 == 0: u = i >> 1
    # If j % 2 == 1: u = (i + 1) >> 1
    
    # Let's calculate (us, vs) and (ut, vt)
    us = (sx >> 1) if sy % 2 == 0 else ((sx + 1) >> 1)
    vs = sy
    ut = (tx >> 1) if ty % 2 == 0 else ((tx + 1) >> 1)
    vt = ty

    # The cost to move between tiles:
    # Moving vertically (changing v) always enters a new tile.
    # Moving horizontally (changing u) might enter a new tile.
    # However, the problem says he can move n units in a direction.
    # If he moves vertically, he crosses |vt - vs| boundaries.
    # If he moves horizontally, he crosses |ut - us| boundaries.
    
    # Wait, the cost is "Each time he enters a tile, he pays a toll of 1".
    # This is equivalent to the L1 distance in the transformed tile-coordinate space,
    # but we must account for the fact that moving vertically might "land" him 
    # in a tile that is the same as the one he was in horizontally.
    
    # Let's re-evaluate:
    # To get from (us, vs) to (ut, vt):
    # Vertical distance: dy = abs(vt - vs)
    # Horizontal distance: dx = abs(ut - us)
    
    # If he moves vertically first, he pays dy. Then he is at (us, vt).
    # Then he moves horizontally to (ut, vt), paying dx.
    # Total = dx + dy.
    
    # But he can optimize. If he is at (us, vs) and wants to go to (ut, vt),
    # he can move to a tile that allows him to "skip" a toll.
    # Specifically, if he moves to a tile (u, v) that is the same tile as (u+1, v),
    # he doesn't pay. But the tiles are fixed.
    
    # The actual distance is:
    # Cost = abs(ut - us) + abs(vt - vs)
    # However, there is a special case: if he moves diagonally, he might save 1.
    # If (us, vs) and (ut, vt) are such that he can move to a tile that 
    # bridges the gap.
    
    # Let's use the property:
    # The distance is max(|ut - us|, (abs(ut - us) + abs(vt - vs) + 1) // 2) 
    # No, that's for different movement.
    
    # Correct logic for this specific tiling:
    # The distance is simply abs(ut - us) + abs(vt - vs), 
    # BUT if he moves diagonally, he can potentially save tolls.
    # Actually, the minimum toll is:
    # abs(ut - us) + abs(vt - vs) - (1 if (us+vs)%2 != (ut+vt)%2 else 0)
    # Wait, let's test Sample 1: S(5,0), T(2,5)
    # us = 5 // 2 = 2, vs = 0
    # ut = (2+1) // 2 = 1, vt = 5
    # dist = abs(1-2) + abs(5-0) = 1 + 5 = 6.
    # Sample 1 output is 5. So he saved 1.
    # (us+vs) = 2+0 = 2 (even), (ut+vt) = 1+5 = 6 (even).
    # Both are even, yet he saved 1.
    
    # Let's reconsider:
    # The cost is simply the L1 distance in the (u, v) space, 
    # but you can move diagonally (change both u and v) for the cost of 1 
    # if the tiles are adjacent.
    # In this tiling, tile (u, v) is adjacent to (u, v+1) and (u+1, v).
    # It is also adjacent to (u-1, v+1) or (u+1, v+1) depending on parity.
    # This is exactly the distance metric of a grid where you can move 
    # to any of the 4 neighbors, and some diagonals.
    # The distance is actually:
    # cost = max(abs(ut - us), abs(vt - vs), (abs(ut - us) + abs(vt - vs) + 1) // 2)
    # No, that's not it.
    
    # Let's use the coordinate transformation:
    # The distance is abs(ut - us) + abs(vt - vs), but he can "jump" 
    # diagonally if the parity allows.
    # The correct formula for this specific problem is:
    # ans = abs(ut - us) + abs(vt - vs)
    # if (us + vs) % 2 != (ut + vt) % 2: ans -= 1
    # Let's check Sample 1: us=2, vs=0, ut=1, vt=5. 
    # (2+0)%2 = 0, (1+5)%2 = 0. Parity is same. ans = 6. Still not 5.
    
    # Let's try: ans = abs(ut - us) + abs(vt - vs)
    # If we can move from (u, v) to (u, v+1) and then (u+1, v+1) is the same tile? No.
    # The only way to save is if we move to a tile that is "diagonal".
    # The distance is actually:
    # ans = abs(ut - us) + abs(vt - vs)
    # if (us % 2 == ut % 2) and (vs % 2 == vt % 2): ... 
    
    # Let's use the logic: 
    # The distance is abs(ut - us) + abs(vt - vs), but we can subtract 1 
    # if we can "shortcut" a corner.
    # A shortcut is possible if we change both u and v and the 
    # parity of the tiles allows it.
    # The condition to subtract 1 is: (us + vs) % 2 != (ut + vt) % 2
    # Wait, I checked Sample 1 and it was 0 and 0. 
    # Let me re-calculate us, vs, ut, vt for Sample 1.
    # S(5,0): sy=0 (even) -> us = 5 // 2 = 2, vs = 0.
    # T(2,5): ty=5 (odd) -> ut = (2+1) // 2 = 1, vt = 5.
    # us+vs = 2, ut+vt = 6. Both even.
    
    # Let's try another approach:
    # The distance is abs(ut - us) + abs(vt - vs).
    # We can reduce it by 1 if (us + vs) % 2 != (ut + vt) % 2.
    # Let's re-read: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    # For j=0: (0,0)&(1,0), (2,0)&(3,0), (4,0)&(5,0) -> Tiles u=0, 1, 2.
    # For j=1: (1,1)&(2,1), (3,1)&(4,1), (5,1)&(6,1) -> Tiles u=1, 2, 3.
    # S(5,0) is in tile u=2, v=0.
    # T(2,5) is in tile u=(2+1)//2 = 1, v=5.
    # Distance = |1-2| + |5-0| = 6.
    # To get 5, we must be able to subtract 1.
    # The condition for subtracting 1 is actually:
    # (us + vs) % 2 == (ut + vt) % 2
    # Let's check: (2+0)%2 = 0, (1+5)%2 = 0. They are equal!
    # So if (us+vs)%2 == (ut+vt)%2, ans = dist - 1?
    # Let's check Sample 2: S(3,1), T(4,1)
    # us = (3+1)//2 = 2, vs = 1.
    # ut = (4+1)//2 = 2, vt = 1.
    # dist = 0 + 0 = 0. Correct.
    
    # Let's check the logic:
    # If we move from (u, v) to (u+1, v+1), the cost is 2 (one for v, one for u).
    # But we can move (u, v) -> (u, v+1) -> (u+1, v+1).
    # If the tile at (u, v+1) is the same as the tile at (u+1, v+1), the cost is 1.
    # Tile (u, v+1) and (u+1, v+1) are the same if (u, v+1) is the "left" part of the tile.
    # That happens if (u + (v+1)) is even.
    # So if we can find a path of diagonals, we save.
    # The distance is abs(ut - us) + abs(vt - vs), and we can save 1 
    # if the parity of (us+vs) and (ut+vt) allows a shortcut.
    # The correct condition is: if (us + vs) % 2 == (ut + vt) % 2, we can save 1,
    # PROVIDED that we actually moved in both dimensions (dx > 0 and dy > 0).
    # Wait, if dx=0 or dy=0, we can't save.
    # Let's refine:
    # dist = abs(ut - us) + abs(vt - vs)
    # if dx > 0 and dy > 0 and (us + vs) % 2 == (ut + vt) % 2: dist -= 1
    # But we can save more than 1? No, because we can only save 1 per 
    # "diagonal" step, and a diagonal step covers 1u and 1v.
    # The number of diagonal steps is min(dx, dy).
    # Each diagonal step saves 1 toll.
    # So the cost is dx + dy - min(dx, dy) = max(dx, dy)?
    # Let's check Sample 1: dx=1, dy=5. max(1, 5) = 5. Correct!
    # Let's check Sample 2: dx=0, dy=0. max(0, 0) = 0. Correct!
    # Let's check if this holds:
    # If we move from (u, v) to (u+1, v+1), we can do it in 2 tolls.
    # But if (u + v+1) is even, then tile (u, v+1) and (u+1, v+1) are the same.
    # So (u, v) -> (u, v+1) [toll 1] -> (u+1, v+1) [