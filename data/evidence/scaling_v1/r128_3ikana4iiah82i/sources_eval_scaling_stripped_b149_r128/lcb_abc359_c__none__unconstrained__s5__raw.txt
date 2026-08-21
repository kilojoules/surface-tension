```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 blocks.
    # In row j, if j is even: tiles are [0,1], [2,3], ... (starts at even i)
    # In row j, if j is odd: tiles are [-1,0], [1,2], ... (starts at odd i)
    # Essentially, in row j, a tile boundary exists between i and i+1 if (i+j) is odd.
    
    # Let's define the cost to move between two points.
    # Moving horizontally in row j:
    # We cross a boundary between i and i+1 if (i+j) is odd.
    # The number of odd (i+j) for i from min(sx, tx) to max(sx, tx)-1.
    # This is equivalent to counting i in range such that i % 2 != j % 2.
    
    # Moving vertically in column i:
    # We cross a boundary between j and j+1 always, because tiles are only 2x1 (horizontal).
    # Every vertical move from j to j+1 enters a new tile.
    # The number of boundaries is simply abs(sy - ty).

    # However, the problem allows choosing the path. 
    # We can move to any column i, then move vertically, then to column tx.
    # But since vertical moves always cost 1 and horizontal moves cost 0 or 1,
    # the optimal strategy is to move to a column i where the horizontal 
    # transitions are minimized, or simply realize that the vertical cost is fixed.
    
    # Wait, the rule says: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    # This means in row j, the boundaries are at i where i+j is odd.
    # If we are at (sx, sy) and move to (tx, ty):
    # 1. We must pay abs(sy - ty) for the vertical transitions.
    # 2. We can pick any row k to perform the horizontal transition from sx to tx.
    # The cost of horizontal transition in row k is the number of i in [min(sx, tx), max(sx, tx)-1]
    # such that i+k is odd.
    
    # Let L = min(sx, tx), R = max(sx, tx).
    # The number of i in [L, R-1] is (R - L).
    # The number of i such that i+k is odd is:
    # If (R-L) is even, there are (R-L)//2 odds and (R-L)//2 evens.
    # If (R-L) is odd, there are either (R-L)//2 or (R-L)//2 + 1 odds.
    # We can choose k to minimize this.
    
    # If R-L is even, the cost is always (R-L)//2.
    # If R-L is odd, we can choose k such that the cost is (R-L)//2.
    # Actually, if R-L > 0, the minimum horizontal cost is (R-L)//2.
    # But we can only change k by moving vertically.
    # The total cost is abs(sy - ty) + (horizontal cost in some row k).
    # We can pick k = sy or k = ty or any row in between.
    # The horizontal cost in row k is the count of i in [L, R-1] where i+k is odd.
    
    # Let's refine:
    # Total Cost = abs(sy - ty) + min(
    #    count_odd(i+sy for i in range(L, R)),
    #    count_odd(i+ty for i in range(L, R))
    # )
    # Actually, we can pick ANY k between sy and ty. 
    # If abs(sy - ty) > 0, we can pick k to be either sy or sy+1.
    # One of these will likely minimize the horizontal cost.
    
    # Let's use a helper to count odds in range [L, R-1] for a given k.
    # The number of i in [L, R-1] such that i+k is odd:
    # This is the number of i in [L, R-1] such that i % 2 != k % 2.
    
    def get_horiz_cost(l, r, k):
        if l == r: return 0
        # Count i in [l, r-1] such that i % 2 != k % 2
        # Total elements = r - l
        # If r-l is even, exactly half are odd, half are even.
        if (r - l) % 2 == 0:
            return (r - l) // 2
        # If r-l is odd, there's one more of one parity.
        # The parity that appears (r-l)//2 + 1 times is the parity of l.
        # We want to minimize the count of i where i % 2 != k % 2.
        # So we want k % 2 to be the same as the majority parity (l % 2).
        # If k % 2 == l % 2, the cost is (r - l) // 2.
        # If k % 2 != l % 2, the cost is (r - l) // 2 + 1.
        return (r - l) // 2 if k % 2 == l % 2 else (r - l) // 2 + 1

    l, r = min(sx, tx), max(sx, tx)
    v_dist = abs(sy - ty)
    
    # We can choose any k in the range [min(sy, ty), max(sy, ty)].
    # We want to minimize get_horiz_cost(l, r, k).
    # The function get_horiz_cost depends only on k % 2.
    # If v_dist == 0, k can only be sy.
    # If v_dist > 0, k can be both even and odd.
    
    if v_dist == 0:
        ans = get_horiz_cost(l, r, sy)
    else:
        # We can pick k such that k % 2 == l % 2 to get the minimum (r-l)//2.
        # Since v_dist > 0, we have at least two different k values (sy, sy+1),
        # one of which must have the parity of l.
        ans = v_dist + (r - l) // 2
        
    # Wait, the vertical cost is always paid. The horizontal cost is added.
    # If v_dist == 0, the answer is just the horizontal cost.
    # If v_dist > 0, the answer is v_dist + (r-l)//2.
    # Let's double check: if v_dist > 0, we can always pick a row k 
    # (either sy or sy+1) that makes the horizontal cost (r-l)//2.
    
    # Final logic:
    # If sy == ty: result is get_horiz_cost(l, r, sy)
    # If sy != ty: result is abs(sy - ty) + (r - l) // 2
    
    # However, the starting tile might be the same as the first horizontal move.
    # "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # If we move horizontally first, and we are in the same tile, cost is 0.
    # The get_horiz_cost logic:
    # If we are at (sx, sy) and move to (tx, sy), we cross boundaries.
    # The number of boundaries is the number of i in [min(sx, tx), max(sx, tx)-1]
    # such that i+sy is odd.
    # This is exactly what get_horiz_cost(l, r, sy) computes.
    
    # Let's re-evaluate the v_dist > 0 case.
    # We can move from (sx, sy) -> (sx, k) -> (tx, k) -> (tx, ty).
    # Cost = abs(sy - k) + get_horiz_cost(l, r, k) + abs(ty - k).
    # We know abs(sy - k) + abs(ty - k) is minimized when k is between sy and ty,
    # in which case it equals abs(sy - ty).
    # Then we minimize get_horiz_cost(l, r, k) for k in [min(sy, ty), max(sy, ty)].
    # If sy != ty, we can pick k to be either parity, so we get (r-l)//2.
    # If sy == ty, k must be sy, so we get get_horiz_cost(l, r, sy).
    
    # One detail: the vertical move itself enters a new tile.
    # Moving from (sx, sy) to (sx, sy+1) enters a new tile. Cost 1.
    # This is already covered by abs(sy - ty).
    
    # Final formula:
    # If sy == ty: return get_horiz_cost(l, r, sy)
    # Else: return abs(sy - ty) + (r - l) // 2
    
    # Let's check Sample 1: 5 0, 2 5
    # l=2, r=5, sy=0, ty=5.
    # sy != ty, so abs(0-5) + (5-2)//2 = 5 + 3//2 = 5 + 1 = 6.
    # Wait, Sample 1 output is 5. Let's re-read.
    # "Move left by 1. Pay a toll of 0."
    # (5, 0) is in tile A_{4,0} U A_{5,0} because 4+0 is even.
    # Moving left to (4, 0) stays in the same tile.
    # Then move up to (4, 1). Enters tile A_{4,1} U A_{5,1} (since 4+1 is odd, A_{4,1} is a tile).
    # Wait, the rule is: "When i+j is even, A_{i,j} and A_{i+1,j} are the same tile."
    # Row 0: (0,1), (2,3), (4,5) are tiles.
    # Row 1: (1,2), (3,4), (5,6) are tiles.
    # Row 2: (0,1), (2,3), (4,5) are tiles.
    # ...
    # Sample 1: S=(5,0), T=(2,5)
    # (5,0) is in tile {A_{4,0}, A_{5,0}}.
    # Move left to (4,0): still in {A_{4,0}, A_{5,0}}. Cost 0.
    # Move up to (4,1): enters tile {A_{3,1}, A_{4,1}}. Cost 1.
    # Move left to (3,1): still in {A_{3,1}, A_{4,1}}. Cost 0.
    # Move up to (3,2): enters tile {A_{2,2}, A_{3,2}}. Cost 1.
    # ... and so on.
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # We must change the row abs(sy - ty) times. Each such move costs 1.
    # Additionally, we might need to move horizontally.
    # But we can "absorb" the horizontal movement into the vertical moves.
    # In each row j, we are in a tile that covers two x-coordinates.
    # The tiles in row j are {(0,1), (2,3), ...} if j is even
    # and {(-1,0), (1,2), (3,4), ...} if j is odd.
    
    # Let's use the property: we can move to any x in the current tile for free.
    # In row j, if we are at x, we can reach x' for free if they are in the same tile.
    # This means we can reach x' if floor(x/2) == floor(x'/2) when j is even,
    # or floor((x-1)/2) == floor((x'-1)/2) when j is odd