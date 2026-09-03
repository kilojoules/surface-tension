```python
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)
    
    # The tiling pattern is:
    # If (i + j) is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are horizontal 2x1 blocks.
    # The boundaries between tiles in the x-direction are at x = i where (i + j) is odd.
    # The boundaries between tiles in the y-direction are always at y = j.
    
    # Let's transform the coordinates to a system where the grid is regular.
    # Consider the "dual" grid of tiles.
    # A tile can be identified by (X, Y) where:
    # Y = j
    # X = (i + j) // 2 if (i + j) is even, or something similar.
    
    # Let's use the property:
    # To move from (sx, sy) to (tx, ty), we must cross some boundaries.
    # Vertical boundaries are at x = k where (k + y) is odd.
    # Horizontal boundaries are at y = k.
    
    # Let u = x + y and v = x - y.
    # A move in x changes u and v by the same amount.
    # A move in y changes u and v by opposite amounts.
    
    # Actually, a simpler way:
    # Let x' = x, y' = y.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # cost = max(0, ceil((tx - sx)/2)) + max(0, ceil((sx - tx)/2)) ... no.
    
    # Let's use the coordinate transformation:
    # X = x + y
    # Y = x - y
    # This is a known problem where the distance is L1 distance in a rotated grid.
    # The minimum toll is:
    # cost = max(0, (tx + ty + 1)//2 - (sx + sy + 1)//2) + 
    #        max(0, (sx - sy + 1)//2 - (tx - ty + 1)//2)
    # Wait, the standard formula for this specific tiling problem is:
    # Let a = (sx + sy), b = (sx - sy), c = (tx + ty), d = (tx - ty)
    # The distance is max(0, (c+1)//2 - (a+1)//2) + max(0, (b+1)//2 - (d+1)//2) 
    # but we can move in 4 directions.
    
    # Correct logic:
    # Let u = x + y and v = x - y.
    # Moving right (x+1): u -> u+1, v -> v+1.
    # Moving left (x-1): u -> u-1, v -> v-1.
    # Moving up (y+1): u -> u+1, v -> v-1.
    # Moving down (y-1): u -> u-1, v -> v+1.
    
    # The boundaries are:
    # Vertical: x = k, k+y is odd => x+y = k+y is odd.
    # Horizontal: y = k => x+y - (x-y) = 2k => x+y and x-y have same parity.
    
    # Let U = (x + y + 1) // 2 and V = (x - y + 1) // 2.
    # Moving right: (x+1, y) -> U' = (x+y+2)//2, V' = (x-y+2)//2.
    # If x+y is even, U increases. If x-y is even, V increases.
    
    # The minimum cost is actually:
    # cost = max(0, (tx + ty + 1)//2 - (sx + sy + 1)//2) + 
    #        max(0, (sx - sy + 1)//2 - (tx - ty + 1)//2)
    # But we can move in any direction. The distance is:
    # dist = max(0, (tx+ty+1)//2 - (sx+sy+1)//2, (sx-sy+1)//2 - (tx-ty+1)//2, ...)
    
    # Let's use the property:
    # Cost = max(0, (tx+ty+1)//2 - (sx+sy+1)//2) + max(0, (sx-sy+1)//2 - (tx-ty+1)//2)
    # This is for moving only Right and Up.
    # For general directions, the cost is:
    # Let A = (sx + sy + 1) // 2, B = (sx - sy + 1) // 2
    # Let C = (tx + ty + 1) // 2, D = (tx - ty + 1) // 2
    # Cost = max(0, C - A, B - D) + max(0, A - C, D - B)
    # Which simplifies to:
    # Cost = max(C - A, 0, D - B, (C - A) + (D - B)) ... no.
    
    # The correct distance in this specific metric is:
    # cost = max(0, C - A) + max(0, B - D) if we only move +x, -y
    # The general formula for the distance between (A, B) and (C, D) 
    # where you can change A or B independently (by moving in y or x) is:
    # cost = max(abs(C - A), abs(D - B))
    
    # Let's verify with Sample 1: 5 0 to 2 5
    # S: x=5, y=0 => A = (5+0+1)//2 = 3, B = (5-0+1)//2 = 3
    # T: x=2, y=5 => C = (2+5+1)//2 = 4, D = (2-5+1)//2 = -1
    # max(abs(4-3), abs(-1-3)) = max(1, 4) = 4. 
    # Sample 1 output is 5. My formula is wrong.
    
    # Let's re-evaluate.
    # To change A by 1, we can move x+1 or y+1.
    # To change B by 1, we can move x+1 or y-1.
    # To increase A and B: move x+1.
    # To increase A and decrease B: move y+1.
    # To decrease A and increase B: move y-1.
    # To decrease A and decrease B: move x-1.
    
    # This is exactly the L1 distance in the (A, B) coordinate system if we could 
    # only move in directions that change A and B.
    # But we can change A and B simultaneously.
    # Actually, the cost is:
    # cost = (abs(C - A) + abs(D - B) + 1) // 2
    # Let's check Sample 1: A=3, B=3, C=4, D=-1.
    # cost = (abs(4-3) + abs(-1-3) + 1) // 2 = (1 + 4 + 1) // 2 = 3. Still wrong.
    
    # Let's use the property:
    # Cost = max(0, C-A) + max(0, D-B) + max(0, A-C) + max(0, B-D) is L1.
    # The moves are:
    # (x+1, y): A -> A or A+1, B -> B or B+1
    # (x-1, y): A -> A or A-1, B -> B or B-1
    # (x, y+1): A -> A or A+1, B -> B or B-1
    # (x, y-1): A -> A or A-1, B -> B or B+1
    
    # Correct logic:
    # To get from (A, B) to (C, D):
    # Let dA = C - A, dB = D - B.
    # We want to find min moves.
    # Each move changes A by {0, 1} and B by {0, 1} etc.
    # Actually, the cost is simply:
    # cost = max(0, C-A) + max(0, B-D) if we only move in +x and -y.
    # The general formula is:
    # cost = max(0, C-A, D-B, (C-A + D-B + 1)//2) ... no.
    
    # Let's use the most reliable formula for this problem:
    # cost = max(0, C-A) + max(0, D-B) where C, D are target and A, B are start.
    # But we can shift the origin.
    # The distance is:
    # cost = max(0, C-A) + max(0, D-B) + max(0, A-C) + max(0, B-D)
    # No, that's L1.
    
    # Let's use: cost = max(0, C-A) + max(0, D-B) 
    # where we can pick which coordinate is A, B and which is C, D.
    # The minimum cost to move between (A, B) and (C, D) is:
    # cost = max(0, C-A) + max(0, D-B) if we can only move in +x and -y.
    # To allow all directions, we can mirror the plane.
    # The distance is:
    # cost = max(0, C-A, D-B, (C-A + D-B + 1)//2) - min(0, C-A, D-B, (C-A + D-B - 1)//2)
    # This is getting complex. Let's use the property:
    # cost = max(0, C-A) + max(0, D-B) is the cost to increase A and decrease B.
    # The 4 possible combinations of (dA, dB) signs give:
    # 1. dA>0, dB>0: cost = max(dA, dB)
    # 2. dA>0, dB<0: cost = dA + abs(dB)
    # 3. dA<0, dB>0: cost = abs(dA) + dB
    # 4. dA<0, dB<0: cost = max(abs(dA), abs(dB))
    
    # Let's test Sample 1: A=3, B=3, C=4, D=-1.
    # dA = 1, dB = -4.
    # Case 2: cost = 1 + 4 = 5. Correct!
    # Sample 2: 3 1 to 4 1.
    # S: A=(3+1+1)//2=2, B=(3-1+1)//2=1.
    # T: A=(4+1+1)//2=3, B=(4-1+1)//2=2.
    # dA = 1, dB = 1.
    # Case 1: cost = max(1, 1) = 1. 
    # Wait, Sample 2 output is 0.
    
    # Re-reading: "When i+j is even, A_{i,j} and A_{i+1,j} are in the same tile."
    # This means the boundary is at x = i + 0.5 where i+j is odd.
    # Let's use: A = (x + y) // 2, B = (x - y + 1) // 2.
    # Sample 2: S(3,1) -> A=2, B=1. T(4,1) -> A=2, B=2.
    # dA=0, dB=1. Case 3: 0+1 = 1. Still not 0.
    
    # Let's use the most basic logic:
    # To move from (sx, sy) to (tx, ty), the cost is:
    # cost = max(0, (tx+ty+1)//2 - (sx+sy+1)//2) + max(0, (sx-sy+1)//2 - (tx-ty+1)//2)
    # But we can move in any direction.
    # The cost is the sum of max(0, delta) for the two independent axes of the tiling.
    # The axes are u = x+y and v = x-y