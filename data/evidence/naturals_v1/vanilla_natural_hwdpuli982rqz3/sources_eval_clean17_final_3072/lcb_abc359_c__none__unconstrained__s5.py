import sys

def solve():
    # Read input from stdin
    # The input consists of two lines: Sx Sy and Tx Ty
    # Using split() handles all whitespace
    data = sys.stdin.read().split()
    if not data:
        return
    
    sx, sy = map(int, data[0:2])
    tx, ty = map(int, data[2:4])

    # The tiling pattern is based on the parity of i + j.
    # When i + j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the x-direction boundaries of tiles are at 
    # x = k where k is even if j is even, and k is odd if j is odd.
    # Essentially, a boundary exists at x = k if k % 2 != j % 2.
    
    # Let's transform the coordinates to a system where we can calculate 
    # the number of boundaries crossed.
    # The number of vertical boundaries crossed when moving from sx to tx at height sy:
    # A boundary is crossed at x = k if k % 2 != sy % 2.
    # The number of such k between min(sx, tx) and max(sx, tx) is 
    # ceil( (max - min) / 2 ) if the start/end are aligned specifically.
    
    # A more robust way:
    # Let f(x, y) be the number of vertical boundaries crossed to reach (x, y) from (0, 0)
    # and g(x, y) be the number of horizontal boundaries.
    # However, the cost is path-dependent.
    
    # Let's use the property:
    # To move from (sx, sy) to (tx, ty), we must change x by dx = |sx - tx| and y by dy = |sy - ty|.
    # Each unit move in y always crosses a horizontal boundary (since tiles are 2x1).
    # Wait, the tiles are 2x1. This means they are joined horizontally.
    # A tile consists of A_{i,j} and A_{i+1,j} if i+j is even.
    # This means for a fixed j, the boundaries are at x = ..., -2, 0, 2, ... if j is even
    # and x = ..., -1, 1, 3, ... if j is odd.
    
    # Let's define the cost to move from (sx, sy) to (tx, ty).
    # The number of horizontal boundaries crossed is simply |sy - ty|.
    # The number of vertical boundaries crossed depends on the path.
    # To minimize cost, we want to move in x at a height j where the number of 
    # boundaries between sx and tx is minimized.
    # For a fixed j, the number of vertical boundaries between sx and tx is:
    # count {k : min(sx, tx) < k <= max(sx, tx) and k % 2 != j % 2}
    # This count is always either floor(|sx-tx|/2) or ceil(|sx-tx|/2).
    
    # Specifically, if we are at height j, the boundaries are at k where k % 2 != j % 2.
    # If we move from sx to tx, we cross (max(sx, tx) - min(sx, tx) + 1) // 2 boundaries
    # if the parity of the boundary matches the parity of the "inner" coordinate.
    
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # The cost is dy + (cost to move dx).
    # The cost to move dx is 0 if dx == 0.
    # If dx > 0, we can pick j = sy or j = ty or any j in between.
    # The number of vertical boundaries crossed at height j is:
    # (dx + 1) // 2 if (min(sx, tx) + 1) % 2 != j % 2 else dx // 2.
    
    # We want to minimize this over j \in [min(sy, ty), max(sy, ty)].
    # If dy > 0, we can pick j to be either sy or ty (or anything in between).
    # One of these will have j % 2 == (min(sx, tx) + 1) % 2, making the cost dx // 2.
    # If dy == 0, we are stuck with j = sy, and the cost is (dx + 1) // 2 if (min(sx, tx) + 1) % 2 != sy % 2 else dx // 2.
    
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    
    # If we can change the parity of j, we can always achieve the floor(dx/2).
    # We can change parity if dy > 0.
    
    # Let's re-evaluate:
    # At height j, boundaries are at k where k % 2 != j % 2.
    # Number of k in (min(sx, tx), max(sx, tx)] is dx.
    # Half of these are even, half are odd.
    # If dx is even, dx/2 boundaries are crossed regardless of j.
    # If dx is odd, (dx+1)//2 boundaries are crossed if j % 2 != (min(sx, tx)+1) % 2,
    # and dx//2 boundaries are crossed if j % 2 == (min(sx, tx)+1) % 2.
    
    # If dy > 0, we can pick j such that j % 2 == (min(sx, tx)+1) % 2.
    # Then vertical cost is dx // 2.
    # If dy == 0, vertical cost is (dx + 1) // 2 if (min(sx, tx)+1) % 2 != sy % 2 else dx // 2.
    
    # Total cost = dy + vertical_cost
    
    # Correcting the dy == 0 case:
    # If dy == 0, j is fixed at sy.
    # Vertical boundaries are k such that k % 2 != sy % 2.
    # Let L = min(sx, tx), R = max(sx, tx).
    # We count k \in {L+1, ..., R} such that k % 2 != sy % 2.
    # This is (R // 2 - L // 2) if R%2 == sy%2 else (R+1)//2 - L//2 ... 
    # Simpler: the number of integers in (L, R] with parity opposite to sy.
    # Total integers is dx. Number of odds is (R+1)//2 - (L+1)//2. Number of evens is R//2 - L//2.
    
    # Let's use a simpler logic for dy == 0:
    # If dy == 0:
    #    cost = dx // 2 if (sx % 2 == sy % 2 and tx % 2 == sy % 2) or (sx % 2 != sy % 2 and tx % 2 != sy % 2) else (dx + 1) // 2
    # Actually, if dy == 0, we cross a boundary at every k where k % 2 != sy % 2.
    # The number of such k in (min(sx, tx), max(sx, tx)] is:
    # ((max(sx, tx) + (sy % 2)) // 2) - ((min(sx, tx) + (sy % 2)) // 2)
    
    # If dy > 0:
    #    cost = dy + dx // 2
    
    # Let's test Sample 1: 5 0, 2 5 -> dx=3, dy=5. Cost = 5 + 3//2 = 5 + 1 = 6? 
    # Sample 1 output is 5. Let's re-read.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Moving from A_{i,j} to A_{i+1,j}: toll 1 if they are different tiles, 0 if same.
    # A_{i,j} and A_{i+1,j} are same if i+j is even.
    # So boundary is at x = i+1 where i+j is odd.
    # Boundary at x = k where (k-1)+j is odd => k+j is even => k % 2 == j % 2.
    
    # Let's re-calculate:
    # Vertical boundaries: x = k where k % 2 == j % 2.
    # Horizontal boundaries: y = k where A_{i, k-1} and A_{i, k} are different.
    # But the rule says "When i+j is even, A_{i,j} and A_{i+1,j} are same".
    # This means A_{i,j} and A_{i,j+1} are ALWAYS different.
    # So every move in y crosses a boundary.
    
    # Vertical cost at height j:
    # Count k \in (min(sx, tx), max(sx, tx)] such that k % 2 == j % 2.
    # This count is (max(sx, tx) + (1 - j%2)) // 2 - (min(sx, tx) + (1 - j%2)) // 2.
    # Wait, if j%2 == 0, we count evens. If j%2 == 1, we count odds.
    # Count of k \in (L, R] with k%2 == p is:
    # (R + (1-p)) // 2 - (L + (1-p)) // 2  -- No.
    # Let's use: count of k <= N with k%2 == 0 is N // 2.
    # Count of k <= N with k%2 == 1 is (N + 1) // 2.
    
    # Let L = min(sx, tx), R = max(sx, tx).
    # If j is even, count is R // 2 - L // 2.
    # If j is odd, count is (R + 1) // 2 - (L + 1) // 2.
    
    # If dy > 0, we can pick j to be even or odd.
    # Min vertical cost = min(R // 2 - L // 2, (R + 1) // 2 - (L + 1) // 2).
    # This is always dx // 2.
    
    # If dy == 0, j is fixed at sy.
    # Vertical cost = (R // 2 - L // 2) if sy % 2 == 0 else ((R + 1) // 2 - (L + 1) // 2).
    
    L, R = min(sx, tx), max(sx, tx)
    dx = R - L
    dy = abs(sy - ty)
    
    if dy > 0:
        ans = dy + (dx // 2)
    else:
        # sy == ty
        vertical_cost = (R // 2 - L // 2) if sy % 2 == 0 else ((R + 1) // 2 - (L + 1) // 2)
        ans = vertical_cost
        
    print(ans)

if __name__ == "__main__":
    solve()